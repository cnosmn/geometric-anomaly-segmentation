"""
Yardımcı fonksiyonlar
"""
import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, List, Optional
import pandas as pd


def rle_encode(mask: np.ndarray) -> str:
    """
    Column-major flattening ile RLE encoding
    
    Args:
        mask: Binary mask (0 ve 255 değerleri)
    
    Returns:
        RLE string (boşluklarla ayrılmış sayılar)
    """
    # Binary threshold
    mask = (mask > 127).astype(np.uint8)
    
    # Column-major flattening (yukarıdan aşağıya, sütun sütun)
    h, w = mask.shape
    
    # Column-major order: her sütunu yukarıdan aşağıya oku
    flattened = []
    for col in range(w):
        for row in range(h):
            flattened.append(mask[row, col])
    
    flattened = np.array(flattened)
    
    # 1-based indexing için 1 ekliyoruz
    pixels = np.where(flattened == 1)[0] + 1
    
    if len(pixels) == 0:
        return ""
    
    # Run-length encoding
    runs = []
    start = pixels[0]
    length = 1
    
    for i in range(1, len(pixels)):
        if pixels[i] == pixels[i-1] + 1:
            length += 1
        else:
            runs.append(f"{start} {length}")
            start = pixels[i]
            length = 1
    
    runs.append(f"{start} {length}")
    
    return " ".join(runs)


def rle_decode(rle_string: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    RLE string'i maskeye çevir (test için)
    
    Args:
        rle_string: RLE encoded string
        shape: (height, width) tuple
    
    Returns:
        Binary mask
    """
    if rle_string == "":
        return np.zeros(shape, dtype=np.uint8)
    
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    
    pairs = list(map(int, rle_string.split()))
    for i in range(0, len(pairs), 2):
        start = pairs[i] - 1  # 1-based'den 0-based'e çevir
        length = pairs[i + 1]
        mask[start:start + length] = 255
    
    # Column-major'dan geri çevir
    mask = mask.reshape((w, h)).T
    
    return mask


def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Intersection over Union hesapla
    
    Args:
        pred_mask: Tahmin edilen maske
        true_mask: Gerçek maske
    
    Returns:
        IoU skoru (0-1 arası)
    """
    pred_mask = (pred_mask > 127).astype(np.uint8)
    true_mask = (true_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def post_process_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morphological operations ile post-processing
    
    Args:
        mask: Binary mask
        kernel_size: Kernel boyutu
    
    Returns:
        İşlenmiş maske
    """
    mask = (mask > 127).astype(np.uint8) * 255
    
    # Küçük gürültüleri temizle
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def create_submission(predictions: dict, output_file: str):
    """
    Submission CSV dosyası oluştur
    
    Args:
        predictions: {image_name: rle_string} dictionary
        output_file: Output CSV dosya yolu
    """
    data = []
    for img_name, rle_str in predictions.items():
        data.append({'id': img_name, 'segmentation': rle_str})
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Submission dosyası oluşturuldu: {output_file}")


def get_image_mask_pairs(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """
    Görüntü ve maske dosya çiftlerini al
    
    Returns:
        [(img_path, mask_path), ...] listesi
    """
    pairs = []
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        mask_file = img_file.replace('img_', 'mask_')
        mask_path = os.path.join(mask_dir, mask_file)
        
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    
    return pairs

