"""
RLE encoding test scripti
"""
import numpy as np
from utils import rle_encode, rle_decode

# Görev dokümanındaki örnek
# 0  0  1  0
# 1  0  1  0
# 1  0  1  0
# 0  0  0  0

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

# Decode test
decoded = rle_decode(rle, (4, 4))
print(f"\nDecoded mask:")
print(decoded)
print(f"\nOriginal == Decoded: {np.array_equal(mask, decoded)}")


