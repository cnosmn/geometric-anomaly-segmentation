"""
Submission.csv'deki her görüntü için orijinal + maske görselleştirmesi
"""
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2

from config import Config
from utils import rle_decode


def visualize_submission_images(
    submission_file='submission.csv',
    output_dir='submission_visualizations',
    max_images=None
):
    """Submission.csv'deki her görüntü için görselleştirme oluştur"""
    
    config = Config()
    
    # Submission dosyasını oku
    df = pd.read_csv(submission_file)
    
    # Output dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Toplam {len(df)} görüntü işlenecek...")
    if max_images:
        df = df.head(max_images)
        print(f"İlk {max_images} görüntü işlenecek...")
    
    for idx, row in df.iterrows():
        img_name = row['id']
        rle_string = row['segmentation']
        
        # Görüntü yolunu oluştur
        img_path = os.path.join(config.TEST_IMG_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️  Görüntü bulunamadı: {img_path}")
            continue
        
        # Orijinal görüntüyü yükle
        original_img = Image.open(img_path).convert('RGB')
        original_img = np.array(original_img)
        
        # RLE'den maskeyi decode et
        if pd.isna(rle_string) or rle_string == '':
            # Boş maske
            pred_mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
            print(f"⚠️  {img_name}: Boş segmentation")
        else:
            try:
                pred_mask = rle_decode(str(rle_string), (config.IMAGE_SIZE, config.IMAGE_SIZE))
            except Exception as e:
                print(f"⚠️  {img_name}: RLE decode hatası - {e}")
                pred_mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
        
        # Maskeyi RGB'ye çevir (görselleştirme için)
        pred_mask_rgb = np.zeros_like(original_img)
        pred_mask_rgb[:, :, 0] = pred_mask  # Kırmızı kanal
        
        # Overlay: Orijinal görüntü + maske
        overlay = original_img.copy()
        mask_area = pred_mask > 0
        overlay[mask_area] = overlay[mask_area] * 0.6 + np.array([255, 0, 0]) * 0.4  # Kırmızı overlay
        
        # Figure oluştur
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Orijinal görüntü
        axes[0].imshow(original_img)
        axes[0].set_title(f'Orijinal Görüntü\n{img_name}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Tahmin edilen maske
        axes[1].imshow(pred_mask, cmap='gray')
        mask_pixels = (pred_mask > 0).sum()
        mask_percentage = (mask_pixels / pred_mask.size) * 100
        axes[1].set_title(f'Tahmin Edilen Maske\n{mask_pixels} piksel ({mask_percentage:.2f}%)', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Orijinal + Maske)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # RLE bilgisi ekle
        rle_info = f"RLE: {rle_string[:50]}..." if len(str(rle_string)) > 50 else f"RLE: {rle_string}"
        fig.suptitle(f'{img_name} - Anomali Segmentasyonu', fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Kaydet
        output_path = os.path.join(output_dir, f'{img_name.replace(".png", "")}_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 10 == 0:
            print(f"✅ {idx+1}/{len(df)} görüntü işlendi...")
    
    print(f"\n{'='*60}")
    print(f"✅ Tüm görselleştirmeler kaydedildi: {output_dir}/")
    print(f"Toplam {len(df)} görüntü işlendi")
    print(f"{'='*60}")


def create_html_viewer(
    submission_file='submission.csv',
    output_dir='submission_visualizations',
    html_file='submission_viewer.html'
):
    """HTML viewer oluştur - tüm görüntüleri tek sayfada göster"""
    
    df = pd.read_csv(submission_file)
    total_images = len(df)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Submission Görselleştirmeleri</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-card h3 {{
            margin-top: 0;
            color: #555;
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .stats {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Submission Görselleştirmeleri</h1>
        <p style="text-align: center; color: #666;">Toplam {total_images} görüntü</p>
        <div class="image-grid">
"""
    
    for idx, row in df.iterrows():
        img_name = row['id']
        rle_string = row['segmentation']
        
        viz_file = f'{img_name.replace(".png", "")}_visualization.png'
        viz_path = os.path.join(output_dir, viz_file)
        
        if os.path.exists(viz_path):
            mask_pixels = len(str(rle_string).split()) // 2 if pd.notna(rle_string) and rle_string != '' else 0
            html_content += f"""
                <div class="image-card">
                    <h3>{img_name}</h3>
                    <img src="{viz_path}" alt="{img_name}">
                    <div class="stats">
                        RLE çift sayısı: {mask_pixels} | 
                        {'Anomali var' if pd.notna(rle_string) and rle_string != '' else 'Anomali yok'}
                    </div>
                </div>
            """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML viewer oluşturuldu: {html_file}")
    print(f"Tarayıcıda açmak için: xdg-open {html_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Submission görselleştirmesi')
    parser.add_argument('--max_images', type=int, default=None, help='Maksimum görüntü sayısı (None = hepsi)')
    parser.add_argument('--html', action='store_true', help='HTML viewer oluştur')
    
    args = parser.parse_args()
    
    # Görselleştirmeleri oluştur
    visualize_submission_images(
        submission_file='submission.csv',
        output_dir='submission_visualizations',
        max_images=args.max_images
    )
    
    # HTML viewer oluştur (isteğe bağlı)
    if args.html:
        create_html_viewer(
            submission_file='submission.csv',
            output_dir='submission_visualizations',
            html_file='submission_viewer.html'
        )

