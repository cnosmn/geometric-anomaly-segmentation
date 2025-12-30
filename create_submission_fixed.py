"""
Submission.csv'yi visualize_predictions.py ile aynı mantıkla oluştur
(TTA olmadan, aynı preprocessing ile)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from config import Config
from dataset import SegmentationDataset
from models import get_model
from utils import rle_encode, post_process_mask, create_submission


def create_submission_fixed():
    """visualize_predictions.py ile aynı mantıkla submission oluştur"""
    
    config = Config()
    device = torch.device(config.DEVICE)
    
    # Model path'leri belirle
    checkpoint_dir = config.CHECKPOINT_DIR
    model_paths = []
    for fold in range(3):
        model_path = os.path.join(
            checkpoint_dir,
            f'unet_efficientnet-b4_fold{fold}_best.pth'
        )
        if os.path.exists(model_path):
            model_paths.append(('unet', 'efficientnet-b4', model_path))
            print(f"✅ Model bulundu: {model_path}")
    
    if not model_paths:
        print("❌ Model checkpoint'i bulunamadı!")
        return
    
    # Test görüntülerini yükle
    test_image_paths = sorted([
        os.path.join(config.TEST_IMG_DIR, f)
        for f in os.listdir(config.TEST_IMG_DIR)
        if f.endswith('.png')
    ])
    
    print(f"\n{'='*60}")
    print(f"{len(model_paths)} model ile ensemble tahmin yapılıyor...")
    print(f"TTA kullanılmıyor (visualize_predictions.py ile aynı mantık)")
    print(f"Toplam {len(test_image_paths)} görüntü")
    print(f"{'='*60}\n")
    
    # Dataset oluştur (TTA olmadan, visualize_predictions.py gibi)
    test_dataset = SegmentationDataset(
        test_image_paths,
        [None] * len(test_image_paths),
        image_size=config.IMAGE_SIZE,
        augment=False,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # visualize_predictions.py gibi batch_size=1
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
            for images, _ in tqdm(test_loader, desc=f'Predicting ({architecture}+{encoder_name})'):
                images = images.to(device)
                outputs = model(images)
                pred_mask = torch.sigmoid(outputs).cpu().numpy()[0, 0]  # [1, 1, H, W] -> [H, W]
                predictions.append(pred_mask)
        
        all_predictions.append(predictions)
    
    # Ensemble: Ortalama al (visualize_predictions.py gibi)
    ensemble_predictions = []
    for i in range(len(test_image_paths)):
        preds = [p[i] for p in all_predictions]
        ensemble_pred = np.mean(preds, axis=0)
        ensemble_predictions.append(ensemble_pred)
    
    # Submission oluştur
    print(f"\n{'='*60}")
    print("Submission oluşturuluyor...")
    print(f"{'='*60}\n")
    
    submission_dict = {}
    
    for idx, (img_path, pred_mask) in enumerate(tqdm(zip(test_image_paths, ensemble_predictions), total=len(test_image_paths), desc='Creating submission')):
        img_name = os.path.basename(img_path)
        
        # Tahmin edilen maskeyi binary yap (visualize_predictions.py ile aynı)
        pred_binary = (pred_mask > config.RLE_THRESHOLD).astype(np.uint8) * 255
        
        # Post-processing (visualize_predictions.py ile aynı)
        if config.USE_MORPHOLOGY:
            pred_binary = post_process_mask(pred_binary, kernel_size=config.MORPH_KERNEL_SIZE)
        
        # RLE encode
        rle_string = rle_encode(pred_binary)
        submission_dict[img_name] = rle_string
    
    # CSV oluştur
    create_submission(submission_dict, config.SUBMISSION_FILE)
    print(f"\n✅ Submission dosyası oluşturuldu: {config.SUBMISSION_FILE}")
    
    # İstatistikleri göster
    import pandas as pd
    df = pd.read_csv(config.SUBMISSION_FILE)
    empty_count = df['segmentation'].isna().sum()
    filled_count = df['segmentation'].notna().sum()
    
    print(f"\n{'='*60}")
    print("İstatistikler:")
    print(f"  Toplam görüntü: {len(df)}")
    print(f"  Anomali tespit edilen: {filled_count}")
    print(f"  Anomali tespit edilmeyen: {empty_count}")
    print(f"{'='*60}")


if __name__ == '__main__':
    create_submission_fixed()


