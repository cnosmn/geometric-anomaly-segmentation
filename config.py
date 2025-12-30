"""
Konfigürasyon dosyası - Tüm hyperparameter'lar burada
"""
import os

class Config:
    # Veri yolları
    BASE_DIR = '/home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2'
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train/images')
    TRAIN_MASK_DIR = os.path.join(BASE_DIR, 'train/masks')
    TEST_IMG_DIR = os.path.join(BASE_DIR, 'test/images')
    
    # Model ayarları
    IMAGE_SIZE = 512
    BATCH_SIZE = 2  # Küçük GPU için optimize edildi (RTX 3050 Ti: 3.68GB VRAM)
    NUM_EPOCHS = 50  # Artırıldı - daha iyi öğrenme için
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Cross-validation
    N_FOLDS = 3  # İlk test için azaltıldı, tam eğitimde 5 kullanılabilir
    RANDOM_SEED = 42
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10  # 10 epoch boyunca iyileşme yoksa dur
    EARLY_STOPPING_MIN_DELTA = 0.0001  # Minimum iyileşme miktarı
    
    # Encoder seçenekleri (test edilecek)
    # Küçük GPU (3.68GB) için hafif encoder'lar
    ENCODERS = [
        'efficientnet-b0',  # En hafif
        'efficientnet-b2',  # Orta
        'resnet34',  # Hafif alternatif
        # 'efficientnet-b4',  # Çok büyük - küçük GPU için kapatıldı
        # 'efficientnet-b5',  # Çok büyük - küçük GPU için kapatıldı
    ]
    
    # Model mimarileri (test edilecek)
    ARCHITECTURES = [
        'unet',
        'unetplusplus',
        'deeplabv3plus',
        # 'fpn',  # İlk test için kapatıldı
    ]
    
    # Loss fonksiyonları
    USE_DICE_LOSS = True
    USE_FOCAL_LOSS = True
    USE_IOU_LOSS = True  # Yeni eklendi - IoU'yu doğrudan optimize eder
    DICE_WEIGHT = 0.3
    FOCAL_WEIGHT = 0.3
    IOU_WEIGHT = 0.4  # IoU loss'a daha fazla ağırlık
    
    # Data augmentation
    USE_AUGMENTATION = True
    AUG_PROBABILITY = 0.5
    
    # Test Time Augmentation
    USE_TTA = True
    TTA_FLIPS = ['horizontal', 'vertical']
    TTA_ROTATIONS = [90, 180, 270]
    
    # Ensemble
    USE_ENSEMBLE = True
    ENSEMBLE_WEIGHTS = 'auto'  # 'auto' veya manuel ağırlıklar
    
    # Post-processing
    USE_MORPHOLOGY = True
    MORPH_KERNEL_SIZE = 3
    
    # RLE encoding
    RLE_THRESHOLD = 0.3  # Binary threshold (düşürüldü - daha hassas tespit için)
    
    # Device
    try:
        import torch
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        DEVICE = 'cpu'
    
    # Checkpoint ve output
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    SUBMISSION_FILE = os.path.join(BASE_DIR, 'submission.csv')

