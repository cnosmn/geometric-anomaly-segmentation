"""
Model tanımlamaları - Farklı mimarileri test etmek için
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class SegmentationModel(nn.Module):
    """Segmentasyon modeli wrapper"""
    
    def __init__(
        self,
        architecture: str = 'unet',
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: Optional[str] = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None
    ):
        super().__init__()
        
        self.architecture = architecture
        self.encoder_name = encoder_name
        
        # Model mimarisini seç
        if architecture == 'unet':
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'unetplusplus':
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'fpn':
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'linknet':
            self.model = smp.Linknet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'pspnet':
            self.model = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        else:
            raise ValueError(f"Bilinmeyen mimari: {architecture}")
    
    def forward(self, x):
        return self.model(x)


class DiceLoss(nn.Module):
    """Dice Loss - Küçük objeler için iyi"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss - Class imbalance için"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        p_t = torch.exp(-bce)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        
        return focal_loss.mean()


class IoULoss(nn.Module):
    """IoU Loss - Doğrudan IoU metriğini optimize eder"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return (1 - iou).mean()


class CombinedLoss(nn.Module):
    """Dice + Focal + IoU Loss kombinasyonu"""
    
    def __init__(self, dice_weight=0.3, focal_weight=0.3, iou_weight=0.4):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.iou_loss = IoULoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        return (self.dice_weight * dice + 
                self.focal_weight * focal + 
                self.iou_weight * iou)


def get_model(
    architecture: str = 'unet',
    encoder_name: str = 'efficientnet-b4',
    device: str = 'cuda'
) -> nn.Module:
    """Model oluştur ve device'a taşı"""
    model = SegmentationModel(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    )
    model = model.to(device)
    return model


def get_loss_function(use_dice=True, use_focal=True, use_iou=True, 
                      dice_weight=0.3, focal_weight=0.3, iou_weight=0.4):
    """Loss fonksiyonu oluştur"""
    if use_dice and use_focal and use_iou:
        return CombinedLoss(dice_weight=dice_weight, focal_weight=focal_weight, iou_weight=iou_weight)
    elif use_dice and use_focal:
        return CombinedLoss(dice_weight=dice_weight, focal_weight=focal_weight, iou_weight=0.0)
    elif use_dice:
        return DiceLoss()
    elif use_focal:
        return FocalLoss()
    elif use_iou:
        return IoULoss()
    else:
        return nn.BCEWithLogitsLoss()


