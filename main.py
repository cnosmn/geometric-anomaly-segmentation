"""
Ana script - Tüm pipeline'ı çalıştırır
"""
import argparse
import os
import json
from pathlib import Path

from config import Config
from train import Trainer
from inference import predict_test_set


def main():
    parser = argparse.ArgumentParser(description='Segmentasyon Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'both'],
        default='both',
        help='Çalıştırılacak mod: train, predict, veya both'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='JSON dosyasından model konfigürasyonu yükle'
    )
    
    args = parser.parse_args()
    config = Config()
    
    # Model konfigürasyonu yükle (varsa)
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
            # Config'i güncelle
            for key, value in model_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    if args.mode in ['train', 'both']:
        print("="*60)
        print("EĞİTİM BAŞLIYOR")
        print("="*60)
        
        trainer = Trainer(config)
        results = trainer.cross_validate()
        
        print("\nCross-validation tamamlandı!")
        print(f"Sonuçlar kaydedildi: {os.path.join(config.OUTPUT_DIR, 'cv_results.json')}")
    
    if args.mode in ['predict', 'both']:
        print("\n" + "="*60)
        print("TAHMIN BAŞLIYOR")
        print("="*60)
        
        # En iyi modelleri yükle
        cv_results_file = os.path.join(config.OUTPUT_DIR, 'cv_results.json')
        
        if os.path.exists(cv_results_file):
            with open(cv_results_file, 'r') as f:
                results = json.load(f)
            
            # En iyi modelleri bul (her fold için)
            model_paths = []
            checkpoint_dir = config.CHECKPOINT_DIR
            
            # Tüm fold'ların en iyi modellerini al
            for model_name, result in results.items():
                architecture, encoder_name = model_name.split('_', 1)
                for fold in range(config.N_FOLDS):
                    model_path = os.path.join(
                        checkpoint_dir,
                        f'{architecture}_{encoder_name}_fold{fold}_best.pth'
                    )
                    if os.path.exists(model_path):
                        model_paths.append((architecture, encoder_name, model_path))
            
            if model_paths:
                print(f"{len(model_paths)} model bulundu. Ensemble tahmin yapılıyor...")
                predict_test_set(
                    config,
                    model_paths=model_paths,
                    use_tta=config.USE_TTA,
                    use_ensemble=config.USE_ENSEMBLE
                )
            else:
                print("Model checkpoint'leri bulunamadı! Önce eğitim yapın.")
        else:
            print("Cross-validation sonuçları bulunamadı! Önce eğitim yapın.")


if __name__ == '__main__':
    main()


