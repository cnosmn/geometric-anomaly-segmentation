"""
Eğitim sonuçlarını gösteren script
TensorBoard loglarından ve checkpoint'lerden bilgi çıkarır
"""
import os
import json
import torch
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

from config import Config


def read_tensorboard_logs(log_dir):
    """TensorBoard loglarından sonuçları okur"""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # IoU değerlerini al
        if 'IoU/Val' in ea.Tags()['scalars']:
            iou_scalars = ea.Scalars('IoU/Val')
            if iou_scalars:
                # En yüksek IoU'yu bul
                max_iou = max([s.value for s in iou_scalars])
                max_epoch = max([s.step for s in iou_scalars if s.value == max_iou])
                return {
                    'max_iou': max_iou,
                    'max_epoch': int(max_epoch),
                    'final_iou': iou_scalars[-1].value if iou_scalars else None,
                    'all_ious': [s.value for s in iou_scalars]
                }
    except Exception as e:
        print(f"  TensorBoard log okunamadı: {e}")
    
    return None


def read_checkpoint_info(checkpoint_path):
    """Checkpoint dosyasından bilgi çıkarır"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'val_iou' in checkpoint:
            return {
                'val_iou': checkpoint['val_iou'],
                'epoch': checkpoint.get('epoch', 'N/A')
            }
    except Exception as e:
        print(f"  Checkpoint okunamadı: {e}")
    
    return None


def show_training_results():
    """Tüm eğitim sonuçlarını gösterir"""
    config = Config()
    
    print("="*80)
    print("EĞİTİM SONUÇLARI RAPORU")
    print("="*80)
    
    # 1. JSON sonuçları kontrol et
    cv_results_file = os.path.join(config.OUTPUT_DIR, 'cv_results.json')
    if os.path.exists(cv_results_file):
        print("\n📊 Cross-Validation Sonuçları (JSON):")
        print("-" * 80)
        with open(cv_results_file, 'r') as f:
            results = json.load(f)
        
        # Sonuçları sırala
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['mean_iou'],
            reverse=True
        )
        
        for model_name, result in sorted_results:
            print(f"\n  Model: {model_name}")
            print(f"    Mean IoU: {result['mean_iou']:.4f} ± {result['std_iou']:.4f}")
            print(f"    Fold IoUs: {[f'{iou:.4f}' for iou in result['fold_ious']]}")
        
        # En iyi model
        best_model = sorted_results[0]
        print(f"\n🏆 EN İYİ MODEL: {best_model[0]}")
        print(f"   Mean IoU: {best_model[1]['mean_iou']:.4f} ± {best_model[1]['std_iou']:.4f}")
    else:
        print("\n⚠️  cv_results.json dosyası bulunamadı!")
    
    # 2. Checkpoint'lerden bilgi çıkar
    print("\n" + "="*80)
    print("📁 CHECKPOINT BİLGİLERİ")
    print("="*80)
    
    checkpoint_dir = config.CHECKPOINT_DIR
    if os.path.exists(checkpoint_dir):
        checkpoint_files = list(Path(checkpoint_dir).glob('*_best.pth'))
        
        if checkpoint_files:
            print(f"\n  Toplam {len(checkpoint_files)} checkpoint bulundu:\n")
            
            checkpoint_results = {}
            
            for ckpt_path in sorted(checkpoint_files):
                model_name = ckpt_path.stem.replace('_best', '')
                print(f"  📦 {model_name}")
                
                # Checkpoint'ten bilgi al
                ckpt_info = read_checkpoint_info(str(ckpt_path))
                if ckpt_info:
                    print(f"     Val IoU: {ckpt_info['val_iou']:.4f}")
                    print(f"     Epoch: {ckpt_info['epoch']}")
                    checkpoint_results[model_name] = ckpt_info
                else:
                    print(f"     (Bilgi çıkarılamadı)")
                
                # Dosya boyutu
                size_mb = ckpt_path.stat().st_size / (1024 * 1024)
                print(f"     Boyut: {size_mb:.2f} MB")
                print()
        else:
            print("  ⚠️  Checkpoint dosyası bulunamadı!")
    else:
        print(f"  ⚠️  Checkpoint dizini bulunamadı: {checkpoint_dir}")
    
    # 3. TensorBoard loglarından bilgi çıkar
    print("="*80)
    print("📈 TENSORBOARD LOGLARI")
    print("="*80)
    
    output_dir = config.OUTPUT_DIR
    if os.path.exists(output_dir):
        log_dirs = [d for d in os.listdir(output_dir) 
                   if os.path.isdir(os.path.join(output_dir, d)) and 'fold' in d]
        
        if log_dirs:
            print(f"\n  Toplam {len(log_dirs)} TensorBoard log dizini bulundu:\n")
            
            tensorboard_results = {}
            
            for log_dir_name in sorted(log_dirs):
                log_dir = os.path.join(output_dir, log_dir_name)
                print(f"  📊 {log_dir_name}")
                
                log_info = read_tensorboard_logs(log_dir)
                if log_info:
                    print(f"     Max IoU: {log_info['max_iou']:.4f} (Epoch {log_info['max_epoch']})")
                    if log_info['final_iou']:
                        print(f"     Final IoU: {log_info['final_iou']:.4f}")
                    tensorboard_results[log_dir_name] = log_info
                else:
                    print(f"     (Log okunamadı)")
                print()
        else:
            print("  ⚠️  TensorBoard log dizini bulunamadı!")
    else:
        print(f"  ⚠️  Output dizini bulunamadı: {output_dir}")
    
    # 4. Özet
    print("="*80)
    print("📋 ÖZET")
    print("="*80)
    
    print("\n  Eğitim tamamlanmış modeller:")
    if os.path.exists(checkpoint_dir):
        checkpoint_files = list(Path(checkpoint_dir).glob('*_best.pth'))
        for ckpt in sorted(checkpoint_files):
            model_name = ckpt.stem.replace('_best', '')
            print(f"    ✓ {model_name}")
    
    print("\n  TensorBoard logları görüntülemek için:")
    print(f"    tensorboard --logdir {output_dir}")
    
    print("\n  JSON sonuçları görüntülemek için:")
    print(f"    cat {cv_results_file}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    show_training_results()

