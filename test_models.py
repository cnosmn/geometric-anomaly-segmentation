"""
Mevcut eğitilmiş modelleri test et ve submission.csv oluştur
"""
import os
from config import Config
from inference import predict_test_set

def test_models():
    """Mevcut checkpoint'leri kullanarak tahmin yap"""
    config = Config()
    
    # Mevcut checkpoint'leri bul
    checkpoint_dir = config.CHECKPOINT_DIR
    model_paths = []
    
    # U-Net + EfficientNet-B4 modellerini bul
    for fold in range(3):
        model_path = os.path.join(
            checkpoint_dir,
            f'unet_efficientnet-b4_fold{fold}_best.pth'
        )
        if os.path.exists(model_path):
            model_paths.append(('unet', 'efficientnet-b4', model_path))
            print(f"✅ Model bulundu: {model_path}")
    
    if not model_paths:
        print("❌ Hiç model checkpoint'i bulunamadı!")
        return
    
    print(f"\n{'='*60}")
    print(f"{len(model_paths)} model ile ensemble tahmin yapılıyor...")
    print(f"Tüm test görüntüleri için tahmin yapılacak (200 görüntü)")
    print(f"{'='*60}\n")
    
    # Tahmin yap - TTA'yı kapatıyoruz çünkü visualize_predictions.py'de iyi sonuçlar verdi
    predict_test_set(
        config,
        model_paths=model_paths,
        use_tta=False,  # TTA'yı kapat - visualize_predictions.py ile aynı mantık
        use_ensemble=config.USE_ENSEMBLE
    )
    
    print(f"\n{'='*60}")
    print("✅ Tahmin tamamlandı!")
    print(f"Submission dosyası: {config.SUBMISSION_FILE}")
    
    # İstatistikleri göster
    import pandas as pd
    df = pd.read_csv(config.SUBMISSION_FILE)
    empty_count = df['segmentation'].isna().sum()
    filled_count = df['segmentation'].notna().sum()
    
    print(f"\nİstatistikler:")
    print(f"  Toplam görüntü: {len(df)}")
    print(f"  Anomali tespit edilen: {filled_count}")
    print(f"  Anomali tespit edilmeyen: {empty_count}")
    print(f"{'='*60}")

if __name__ == '__main__':
    test_models()

