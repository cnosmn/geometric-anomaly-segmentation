"""
Eğitim scripti - Cross-validation ile model karşılaştırma
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

from config import Config
from dataset import SegmentationDataset
from models import get_model, get_loss_function
from utils import calculate_iou, get_image_mask_pairs


class Trainer:
    """Model eğitici"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # GPU bellek optimizasyonu
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()  # GPU cache'i temizle
            # Bellek fragmentasyonunu azalt
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Dizinleri oluştur
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        architecture: str,
        encoder_name: str,
        fold: int
    ):
        """Bir fold için eğitim"""
        
        # GPU bellek temizliği
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Model oluştur
        model = get_model(architecture, encoder_name, self.device)
        
        # Gradient checkpointing (bellek tasarrufu için)
        if hasattr(model.model, 'encoder') and self.device.type == 'cuda':
            try:
                # Bazı encoder'lar için gradient checkpointing
                from torch.utils.checkpoint import checkpoint
            except:
                pass
        
        # Loss ve optimizer
        criterion = get_loss_function(
            use_dice=self.config.USE_DICE_LOSS,
            use_focal=self.config.USE_FOCAL_LOSS,
            use_iou=getattr(self.config, 'USE_IOU_LOSS', True),
            dice_weight=self.config.DICE_WEIGHT,
            focal_weight=self.config.FOCAL_WEIGHT,
            iou_weight=getattr(self.config, 'IOU_WEIGHT', 0.4)
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler - ReduceLROnPlateau (IoU için optimize)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # IoU için 'max'
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
        # Early Stopping
        early_stopping_patience = getattr(self.config, 'EARLY_STOPPING_PATIENCE', 10)
        early_stopping_min_delta = getattr(self.config, 'EARLY_STOPPING_MIN_DELTA', 0.0001)
        early_stopping_counter = 0
        best_iou_for_early_stop = 0.0
        
        # TensorBoard
        writer = SummaryWriter(
            log_dir=os.path.join(self.config.OUTPUT_DIR, f'{architecture}_{encoder_name}_fold{fold}')
        )
        
        best_iou = 0.0
        best_model_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'{architecture}_{encoder_name}_fold{fold}_best.pth'
        )
        
        # Eğitim loop
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}')
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Maske shape kontrolü: [B, 1, H, W] olmalı
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # GPU bellek temizliği (her batch'ten sonra)
                if self.device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_iou = self.validate(model, val_loader)
            
            # Learning rate scheduler (IoU'ya göre)
            scheduler.step(val_iou)
            
            # Log
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('IoU/Val', val_iou, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR', current_lr, epoch)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, LR: {current_lr:.6f}')
            
            # Best model kaydet
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': val_iou,
                    'architecture': architecture,
                    'encoder_name': encoder_name,
                }, best_model_path)
                print(f'Best model kaydedildi! IoU: {best_iou:.4f}')
            
            # Early Stopping kontrolü
            if val_iou > best_iou_for_early_stop + early_stopping_min_delta:
                best_iou_for_early_stop = val_iou
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f'Early stopping! {early_stopping_patience} epoch boyunca iyileşme yok.')
                    print(f'En iyi IoU: {best_iou:.4f}')
                    break
        
        writer.close()
        return best_iou, best_model_path
    
    def validate(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Validation"""
        model.eval()
        ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = model(images)
                pred_masks = torch.sigmoid(outputs).cpu().numpy()
                true_masks = masks.cpu().numpy()
                
                for pred, true in zip(pred_masks, true_masks):
                    pred_binary = (pred[0] > 0.5).astype(np.uint8) * 255
                    true_binary = (true[0] > 0.5).astype(np.uint8) * 255
                    iou = calculate_iou(pred_binary, true_binary)
                    ious.append(iou)
        
        return np.mean(ious)
    
    def cross_validate(self):
        """Cross-validation ile model karşılaştırma"""
        
        # Veri çiftlerini al
        pairs = get_image_mask_pairs(
            self.config.TRAIN_IMG_DIR,
            self.config.TRAIN_MASK_DIR
        )
        
        image_paths = [p[0] for p in pairs]
        mask_paths = [p[1] for p in pairs]
        
        # KFold
        kfold = KFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )
        
        results = {}
        
        # Her mimari ve encoder kombinasyonunu test et
        for architecture in self.config.ARCHITECTURES:
            for encoder_name in self.config.ENCODERS:
                print(f"\n{'='*60}")
                print(f"Testing: {architecture} + {encoder_name}")
                print(f"{'='*60}")
                
                fold_ious = []
                
                for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths)):
                    print(f"\nFold {fold + 1}/{self.config.N_FOLDS}")
                    
                    # DataLoader'ları oluştur
                    train_dataset = SegmentationDataset(
                        [image_paths[i] for i in train_idx],
                        [mask_paths[i] for i in train_idx],
                        image_size=self.config.IMAGE_SIZE,
                        augment=self.config.USE_AUGMENTATION,
                        mode='train'
                    )
                    
                    val_dataset = SegmentationDataset(
                        [image_paths[i] for i in val_idx],
                        [mask_paths[i] for i in val_idx],
                        image_size=self.config.IMAGE_SIZE,
                        augment=False,
                        mode='val'
                    )
                    
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=2,  # Küçük GPU için azaltıldı
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=False  # Bellek tasarrufu
                    )
                    
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.config.BATCH_SIZE,
                        shuffle=False,
                        num_workers=2,  # Küçük GPU için azaltıldı
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=False  # Bellek tasarrufu
                    )
                    
                    # Eğit
                    best_iou, model_path = self.train_fold(
                        train_loader,
                        val_loader,
                        architecture,
                        encoder_name,
                        fold
                    )
                    
                    fold_ious.append(best_iou)
                
                # Ortalama IoU
                mean_iou = np.mean(fold_ious)
                std_iou = np.std(fold_ious)
                
                results[f'{architecture}_{encoder_name}'] = {
                    'mean_iou': mean_iou,
                    'std_iou': std_iou,
                    'fold_ious': fold_ious
                }
                
                print(f"\n{architecture} + {encoder_name}:")
                print(f"  Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
                print(f"  Fold IoUs: {fold_ious}")
        
        # Sonuçları kaydet
        import json
        results_file = os.path.join(self.config.OUTPUT_DIR, 'cv_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # En iyi modeli bul
        best_model = max(results.items(), key=lambda x: x[1]['mean_iou'])
        print(f"\n{'='*60}")
        print(f"EN İYİ MODEL: {best_model[0]}")
        print(f"IoU: {best_model[1]['mean_iou']:.4f} ± {best_model[1]['std_iou']:.4f}")
        print(f"{'='*60}")
        
        return results


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    results = trainer.cross_validate()

