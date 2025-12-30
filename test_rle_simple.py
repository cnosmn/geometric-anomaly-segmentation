"""
RLE encoding test scripti (basit versiyon)
"""
import numpy as np
import sys
sys.path.insert(0, '.')

# RLE encode fonksiyonunu direkt test et
def rle_encode(mask: np.ndarray) -> str:
    """Column-major flattening ile RLE encoding"""
    mask = (mask > 127).astype(np.uint8)
    h, w = mask.shape
    
    # Column-major order: her sütunu yukarıdan aşağıya oku
    flattened = []
    for col in range(w):
        for row in range(h):
            flattened.append(mask[row, col])
    
    flattened = np.array(flattened)
    pixels = np.where(flattened == 1)[0] + 1
    
    if len(pixels) == 0:
        return ""
    
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

# Görev dokümanındaki örnek
mask = np.array([
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
], dtype=np.uint8) * 255

print("Original mask:")
print(mask)
print()

# Column-major flattening manuel kontrol
print("Column-major flattening (yukarıdan aşağıya, sütun sütun):")
for col in range(4):
    print(f"Column {col}: {mask[:, col].tolist()}")

flattened_manual = []
for col in range(4):
    for row in range(4):
        flattened_manual.append(mask[row, col])

print(f"\nFlattened (manual): {flattened_manual}")
print(f"1-based indices where value=255: {[i+1 for i, v in enumerate(flattened_manual) if v == 255]}")

# RLE encode
rle = rle_encode(mask)
print(f"\nRLE encoded: {rle}")
print(f"Expected: 2 2 9 3")

if rle == "2 2 9 3":
    print("✅ RLE encoding DOĞRU!")
else:
    print("❌ RLE encoding YANLIŞ!")


