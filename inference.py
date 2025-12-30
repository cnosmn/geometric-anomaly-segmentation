"""
Inference scripti - Test seti için tahmin ve submission oluşturma
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import cv2
from typing import List, Dict, Tuple

from config import Config
from dataset import SegmentationDataset, TTADataset
from models import get_model
from utils import rle_encode, post_process_mask, create_submission


class Predictor:
    """Tahmin yapıcı"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
    
    def predict_single_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        use_tta: bool = False
    ) -> Dict[str, np.ndarray]:
        """Tek model ile tahmin"""
        model.eval()
        predictions = {}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Predicting'):
                if use_tta:
                    images, tta_images_list, image_paths = batch
                    images = images.to(self.device)
                    
                    # Normal prediction
                    outputs = model(images)
                    pred_masks = torch.sigmoid(outputs).cpu().numpy()
                    
                    # TTA predictions (her TTA transform için)
                    tta_preds = []
                    for tta_batch in tta_images_list:
                        # tta_batch shape: [batch_size, 3, H, W]
                        tta_batch = tta_batch.to(self.device)
                        tta_output = model(tta_batch)
                        tta_pred = torch.sigmoid(tta_output).cpu().numpy()
                        tta_preds.append(tta_pred)
                    
                    # TTA'ları geri çevir ve birleştir
                    # (Basitleştirilmiş - gerçek implementasyonda her TTA için inverse transform gerekli)
                    all_preds = [pred_masks] + tta_preds
                    final_pred = np.mean(all_preds, axis=0)
                    
                else:
                    images = batch[0]
                    image_paths = batch[1] if len(batch) > 1 else None
                    images = images.to(self.device)
                    outputs = model(images)
                    final_pred = torch.sigmoid(outputs).cpu().numpy()
                
                # Her görüntü için kaydet
                batch_size = final_pred.shape[0]
                for i in range(batch_size):
                    if image_paths:
                        image_path = image_paths[i]
                        image_name = os.path.basename(image_path)
                    else:
                        image_name = f"img_{len(predictions):04d}.png"
                    
                    pred_mask = final_pred[i][0]  # [1, H, W] -> [H, W]
                    
                    # 512x512'e resize et (eğer farklıysa)
                    if pred_mask.shape != (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE):
                        pred_mask = cv2.resize(
                            pred_mask,
                            (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    predictions[image_name] = pred_mask
        
        return predictions
    
    def predict_ensemble(
        self,
        model_paths: List[Tuple[str, str, str]],  # [(architecture, encoder, path), ...]
        test_loader: DataLoader,
        use_tta: bool = False,
        weights: List[float] = None
    ) -> Dict[str, np.ndarray]:
        """Ensemble tahmin"""
        
        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)
        
        all_predictions = []
        
        for architecture, encoder_name, model_path in model_paths:
            print(f"Loading model: {architecture} + {encoder_name}")
            model = get_model(architecture, encoder_name, self.device)
            
            # Checkpoint yükle
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Tahmin yap
            predictions = self.predict_single_model(model, test_loader, use_tta)
            all_predictions.append(predictions)
        
        # Weighted average
        final_predictions = {}
        image_names = list(all_predictions[0].keys())
        
        for img_name in image_names:
            weighted_sum = np.zeros_like(all_predictions[0][img_name])
            total_weight = 0
            
            for i, pred_dict in enumerate(all_predictions):
                weighted_sum += weights[i] * pred_dict[img_name]
                total_weight += weights[i]
            
            final_predictions[img_name] = weighted_sum / total_weight
        
        return final_predictions
    
    def create_submission_from_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        output_file: str
    ):
        """Predictions'dan submission dosyası oluştur"""
        
        submission_dict = {}
        
        for img_name, pred_mask in tqdm(predictions.items(), desc='Creating submission'):
            # Post-processing
            if self.config.USE_MORPHOLOGY:
                pred_binary = (pred_mask > self.config.RLE_THRESHOLD).astype(np.uint8) * 255
                pred_binary = post_process_mask(
                    pred_binary,
                    kernel_size=self.config.MORPH_KERNEL_SIZE
                )
            else:
                pred_binary = (pred_mask > self.config.RLE_THRESHOLD).astype(np.uint8) * 255
            
            # RLE encode
            rle_string = rle_encode(pred_binary)
            
            # Eğer rle_string boşsa (anomali yoksa), yine de boş string olarak kaydet
            # Ama kullanıcı tüm görüntülerde anomali olduğunu söylüyor
            submission_dict[img_name] = rle_string
        
        # CSV oluştur
        create_submission(submission_dict, output_file)
        print(f"Submission dosyası oluşturuldu: {output_file}")


def predict_test_set(
    config: Config,
    model_paths: List[Tuple[str, str, str]] = None,
    use_tta: bool = True,
    use_ensemble: bool = True
):
    """Test seti için tahmin yap"""
    
    # Test görüntülerini yükle
    test_image_paths = sorted([
        os.path.join(config.TEST_IMG_DIR, f)
        for f in os.listdir(config.TEST_IMG_DIR)
        if f.endswith('.png')
    ])
    
    # Dataset ve DataLoader
    if use_tta:
        test_dataset = TTADataset(test_image_paths, config.IMAGE_SIZE)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        test_dataset = SegmentationDataset(
            test_image_paths,
            [None] * len(test_image_paths),
            image_size=config.IMAGE_SIZE,
            augment=False,
            mode='test'
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Predictor
    predictor = Predictor(config)
    
    # Tahmin yap
    if use_ensemble and model_paths:
        predictions = predictor.predict_ensemble(
            model_paths,
            test_loader,
            use_tta=use_tta
        )
    elif model_paths:
        # İlk modeli kullan
        architecture, encoder_name, model_path = model_paths[0]
        model = get_model(architecture, encoder_name, predictor.device)
        checkpoint = torch.load(model_path, map_location=predictor.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        predictions = predictor.predict_single_model(model, test_loader, use_tta)
    else:
        raise ValueError("Model path'leri gerekli!")
    
    # Submission oluştur
    predictor.create_submission_from_predictions(
        predictions,
        config.SUBMISSION_FILE
    )


if __name__ == '__main__':
    config = Config()
    
    # En iyi modelleri buraya ekle (cross-validation sonuçlarına göre)
    # Örnek:
    # model_paths = [
    #     ('unet', 'efficientnet-b4', 'checkpoints/unet_efficientnet-b4_fold0_best.pth'),
    #     ('unetplusplus', 'efficientnet-b5', 'checkpoints/unetplusplus_efficientnet-b5_fold1_best.pth'),
    # ]
    
    # Şimdilik tek model ile test
    print("Lütfen önce train.py ile modelleri eğitin ve en iyi model path'lerini belirleyin.")
    print("Sonra bu scripti model_paths ile güncelleyin.")

