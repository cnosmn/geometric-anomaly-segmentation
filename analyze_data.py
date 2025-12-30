from PIL import Image
import numpy as np
import os

# Veri analizi
train_img_dir = 'train/images'
train_mask_dir = 'train/masks'

print("=== VERİ SETİ ANALİZİ ===\n")

# İlk 5 görüntüyü analiz et
imgs = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.png')])[:5]

for img_name in imgs:
    img_path = os.path.join(train_img_dir, img_name)
    mask_name = img_name.replace('img_', 'mask_')
    mask_path = os.path.join(train_mask_dir, mask_name)
    
    if os.path.exists(mask_path):
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        
        anomaly_pixels = (mask_arr > 0).sum()
        total_pixels = mask_arr.size
        coverage = anomaly_pixels / total_pixels * 100
        
        # Görüntü istatistikleri
        img_arr = np.array(img)
        
        print(f"{img_name}:")
        print(f"  Görüntü boyutu: {img.size}, Mod: {img.mode}")
        print(f"  Maske boyutu: {mask.size}, Mod: {mask.mode}")
        print(f"  Anomali piksel sayısı: {anomaly_pixels} ({coverage:.2f}%)")
        print(f"  Maske değerleri: {np.unique(mask_arr)}")
        print()

# Tüm veri seti istatistikleri
all_imgs = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.png')])
coverages = []
has_anomaly = []

for img_name in all_imgs:
    mask_name = img_name.replace('img_', 'mask_')
    mask_path = os.path.join(train_mask_dir, mask_name)
    
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        anomaly_pixels = (mask_arr > 0).sum()
        total_pixels = mask_arr.size
        coverage = anomaly_pixels / total_pixels * 100
        coverages.append(coverage)
        has_anomaly.append(anomaly_pixels > 0)

print(f"\n=== GENEL İSTATİSTİKLER ===")
print(f"Toplam eğitim görüntüsü: {len(all_imgs)}")
print(f"Anomali içeren görüntü sayısı: {sum(has_anomaly)}")
print(f"Anomali içermeyen görüntü sayısı: {len(has_anomaly) - sum(has_anomaly)}")
print(f"Ortalama anomali kapsama: {np.mean(coverages):.2f}%")
print(f"Min anomali kapsama: {np.min(coverages):.2f}%")
print(f"Max anomali kapsama: {np.max(coverages):.2f}%")


