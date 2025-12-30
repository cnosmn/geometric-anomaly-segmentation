"""
Test görüntülerinde tahmin edilen maskeleri görselleştir
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

from config import Config
from dataset import SegmentationDataset
from models import get_model
from utils import post_process_mask


def visualize_predictions(
    num_images=10,
    model_paths=None,
    output_dir='visualizations'
):
    """Test görüntülerinde tahmin edilen maskeleri görselleştir"""
    
    config = Config()
    device = torch.device(config.DEVICE)
    
    # Output dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Model path'leri belirle
    if model_paths is None:
        checkpoint_dir = config.CHECKPOINT_DIR
        model_paths = []
        for fold in range(3):
            model_path = os.path.join(
                checkpoint_dir,
                f'unet_efficientnet-b4_fold{fold}_best.pth'
            )
            if os.path.exists(model_path):
                model_paths.append(('unet', 'efficientnet-b4', model_path))
    
    if not model_paths:
        print("❌ Model checkpoint'i bulunamadı!")
        return
    
    # Test görüntülerini yükle
    test_image_paths = sorted([
        os.path.join(config.TEST_IMG_DIR, f)
        for f in os.listdir(config.TEST_IMG_DIR)
        if f.endswith('.png')
    ])[:num_images]
    
    # Dataset oluştur
    test_dataset = SegmentationDataset(
        test_image_paths,
        [None] * len(test_image_paths),
        image_size=config.IMAGE_SIZE,
        augment=False,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Modelleri yükle ve ensemble tahmin yap
    all_predictions = []
    
    for architecture, encoder_name, model_path in model_paths:
        print(f"Model yükleniyor: {architecture} + {encoder_name}")
        model = get_model(architecture, encoder_name, device)
        
        # Checkpoint yükle
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                pred_mask = torch.sigmoid(outputs).cpu().numpy()[0, 0]  # [1, 1, H, W] -> [H, W]
                predictions.append(pred_mask)
        
        all_predictions.append(predictions)
    
    # Ensemble: Ortalama al
    ensemble_predictions = []
    for i in range(len(test_image_paths)):
        preds = [p[i] for p in all_predictions]
        ensemble_pred = np.mean(preds, axis=0)
        ensemble_predictions.append(ensemble_pred)
    
    # Görselleştir
    print(f"\n{len(test_image_paths)} görüntü görselleştiriliyor...")
    
    for idx, (img_path, pred_mask) in enumerate(zip(test_image_paths, ensemble_predictions)):
        # Orijinal görüntüyü yükle
        original_img = Image.open(img_path).convert('RGB')
        original_img = np.array(original_img)
        
        # Tahmin edilen maskeyi binary yap
        pred_binary = (pred_mask > config.RLE_THRESHOLD).astype(np.uint8) * 255
        
        # Post-processing
        if config.USE_MORPHOLOGY:
            pred_binary = post_process_mask(pred_binary, kernel_size=config.MORPH_KERNEL_SIZE)
        
        # Maskeyi RGB'ye çevir (görselleştirme için)
        pred_mask_rgb = np.zeros_like(original_img)
        pred_mask_rgb[:, :, 0] = pred_binary  # Kırmızı kanal
        
        # Overlay: Orijinal görüntü + maske
        overlay = original_img.copy()
        mask_area = pred_binary > 0
        overlay[mask_area] = overlay[mask_area] * 0.6 + np.array([255, 0, 0]) * 0.4  # Kırmızı overlay
        
        # Figure oluştur
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Orijinal görüntü
        axes[0].imshow(original_img)
        axes[0].set_title(f'Orijinal Görüntü\n{os.path.basename(img_path)}', fontsize=12)
        axes[0].axis('off')
        
        # Tahmin edilen maske
        axes[1].imshow(pred_binary, cmap='gray')
        axes[1].set_title('Tahmin Edilen Maske', fontsize=12)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Orijinal + Maske)', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Kaydet
        output_path = os.path.join(output_dir, f'prediction_{os.path.basename(img_path)}')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {idx+1}/{len(test_image_paths)}: {os.path.basename(img_path)} kaydedildi")
    
    print(f"\n{'='*60}")
    print(f"✅ Tüm görselleştirmeler kaydedildi: {output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=10, help='Görselleştirilecek görüntü sayısı')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output dizini')
    
    args = parser.parse_args()
    
    visualize_predictions(
        num_images=args.num_images,
        output_dir=args.output_dir
    )


