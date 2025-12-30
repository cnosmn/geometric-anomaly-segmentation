"""
PyTorch Dataset sınıfı
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
import os


class SegmentationDataset(Dataset):
    """Segmentasyon için custom dataset"""
    
    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        image_size: int = 512,
        augment: bool = False,
        mode: str = 'train'
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment
        self.mode = mode
        
        # Augmentation pipeline
        if augment and mode == 'train':
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                # A.GaussNoise kaldırıldı (uyarı nedeniyle)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Görüntü yükle
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        
        # Maske yükle (eğer varsa)
        if (self.mask_paths and 
            self.mask_paths[idx] is not None and 
            os.path.exists(self.mask_paths[idx])):
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask = np.array(mask)
        else:
            # Test için boş maske
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Augmentation uygula
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Maske'yi binary yap (0 veya 1) ve [H, W] -> [1, H, W] yap
        mask = (mask > 127).float()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        return image, mask


class TTADataset(Dataset):
    """Test Time Augmentation için dataset"""
    
    def __init__(
        self,
        image_paths: list,
        image_size: int = 512
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        
        # Normal transform
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # TTA transforms
        self.tta_transforms = [
            A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.Rotate(limit=90, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.Rotate(limit=180, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.Rotate(limit=270, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)
        
        # Normal transform
        transformed = self.transform(image=image)
        normal_image = transformed['image']
        
        # TTA transforms
        tta_images = []
        for tta_transform in self.tta_transforms:
            transformed = tta_transform(image=image)
            tta_images.append(transformed['image'])
        
        return normal_image, tta_images, self.image_paths[idx]

