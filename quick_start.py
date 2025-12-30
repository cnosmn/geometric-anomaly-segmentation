"""
Hızlı başlangıç scripti - Sadece en iyi modelleri test etmek için
"""
import os
from config import Config
from train import Trainer
from inference import predict_test_set
import json

def quick_train():
    """Hızlı eğitim - Sadece en iyi kombinasyonları test et"""
    config = Config()
    
    # Sadece en iyi kombinasyonları test et (zaman kazanmak için)
    config.ARCHITECTURES = ['unet', 'unetplusplus', 'deeplabv3plus']
    config.ENCODERS = ['efficientnet-b4', 'efficientnet-b5']
    config.N_FOLDS = 3  # 5 yerine 3 fold (daha hızlı)
    config.NUM_EPOCHS = 30  # 50 yerine 30 epoch
    
    print("Hızlı eğitim modu başlatılıyor...")
    print(f"Test edilecek mimariler: {config.ARCHITECTURES}")
    print(f"Test edilecek encoder'lar: {config.ENCODERS}")
    print(f"Fold sayısı: {config.N_FOLDS}")
    print(f"Epoch sayısı: {config.NUM_EPOCHS}")
    
    trainer = Trainer(config)
    results = trainer.cross_validate()
    
    return results


def quick_predict():
    """En iyi modellerle tahmin yap"""
    config = Config()
    
    cv_results_file = os.path.join(config.OUTPUT_DIR, 'cv_results.json')
    
    if not os.path.exists(cv_results_file):
        print("Önce eğitim yapmalısınız! quick_train() çalıştırın.")
        return
    
    with open(cv_results_file, 'r') as f:
        results = json.load(f)
    
    # En iyi 3 modeli seç
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['mean_iou'],
        reverse=True
    )[:3]
    
    print(f"\nEn iyi 3 model seçildi:")
    for model_name, result in sorted_results:
        print(f"  {model_name}: IoU = {result['mean_iou']:.4f}")
    
    # Model path'lerini topla
    model_paths = []
    checkpoint_dir = config.CHECKPOINT_DIR
    
    for model_name, result in sorted_results:
        architecture, encoder_name = model_name.split('_', 1)
        # En iyi fold'u seç
        best_fold = result['fold_ious'].index(max(result['fold_ious']))
        model_path = os.path.join(
            checkpoint_dir,
            f'{architecture}_{encoder_name}_fold{best_fold}_best.pth'
        )
        if os.path.exists(model_path):
            model_paths.append((architecture, encoder_name, model_path))
    
    if model_paths:
        print(f"\n{len(model_paths)} model ile tahmin yapılıyor...")
        predict_test_set(
            config,
            model_paths=model_paths,
            use_tta=config.USE_TTA,
            use_ensemble=config.USE_ENSEMBLE
        )
    else:
        print("Model checkpoint'leri bulunamadı!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        quick_predict()
    else:
        print("Hızlı eğitim başlatılıyor...")
        print("Sadece tahmin yapmak için: python quick_start.py predict")
        results = quick_train()
        print("\nEğitim tamamlandı! Şimdi tahmin yapabilirsiniz:")
        print("python quick_start.py predict")


