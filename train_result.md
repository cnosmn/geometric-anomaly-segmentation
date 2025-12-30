# 📊 Eğitim Sonuçları Raporu

**Tarih**: 29 Aralık 2024  
**Model**: U-Net + EfficientNet-B4  
**Cross-Validation**: 3-Fold

---

## 🎯 Özet

Bu rapor, Geometric Anomaly Segmentation görevi için eğitilmiş modellerin performans sonuçlarını içermektedir.

### Genel Performans

- **Toplam Model Sayısı**: 3 fold
- **Mimari**: U-Net
- **Encoder**: EfficientNet-B4
- **Ortalama IoU**: **97.65%**
- **En İyi Fold IoU**: **98.06%** (fold2)
- **En Düşük Fold IoU**: **97.34%** (fold1)

---

## 📈 Detaylı Fold Sonuçları

### Fold 0 (unet_efficientnet-b4_fold0)

- **Validation IoU**: **0.9756** (97.56%)
- **En İyi Epoch**: 20
- **Final IoU**: 0.9752 (97.52%)
- **Checkpoint Boyutu**: 226.48 MB
- **Checkpoint Yolu**: `checkpoints/unet_efficientnet-b4_fold0_best.pth`

**Performans Analizi**:
- Fold 0, 20. epoch'ta en iyi performansı göstermiştir
- Final epoch'ta hafif bir düşüş görülmüştür (0.9756 → 0.9752)
- Model stabil bir şekilde yüksek IoU değerlerine ulaşmıştır

---

### Fold 1 (unet_efficientnet-b4_fold1)

- **Validation IoU**: **0.9734** (97.34%)
- **En İyi Epoch**: 29
- **Final IoU**: 0.9734 (97.34%)
- **Checkpoint Boyutu**: 226.48 MB
- **Checkpoint Yolu**: `checkpoints/unet_efficientnet-b4_fold1_best.pth`

**Performans Analizi**:
- Fold 1, 29. epoch'ta (son epoch) en iyi performansı göstermiştir
- Model eğitim süresince sürekli iyileşme göstermiştir
- En düşük IoU değerine sahip fold olmasına rağmen, hala çok yüksek bir performans sergilemiştir

---

### Fold 2 (unet_efficientnet-b4_fold2)

- **Validation IoU**: **0.9806** (98.06%) ⭐ **EN İYİ**
- **En İyi Epoch**: 22
- **Final IoU**: 0.9798 (97.98%)
- **Checkpoint Boyutu**: 226.48 MB
- **Checkpoint Yolu**: `checkpoints/unet_efficientnet-b4_fold2_best.pth`

**Performans Analizi**:
- Fold 2, tüm fold'lar arasında en yüksek IoU değerine sahiptir
- 22. epoch'ta en iyi performansı göstermiştir
- Final epoch'ta hafif bir düşüş görülmüştür (0.9806 → 0.9798)
- Model çok yüksek bir segmentasyon kalitesi elde etmiştir

---

## 📊 İstatistiksel Analiz

### Fold Performans Karşılaştırması

| Fold | IoU | Epoch | Final IoU | Fark |
|------|-----|-------|-----------|------|
| Fold 0 | 0.9756 | 20 | 0.9752 | -0.0004 |
| Fold 1 | 0.9734 | 29 | 0.9734 | 0.0000 |
| Fold 2 | 0.9806 | 22 | 0.9798 | -0.0008 |

### Özet İstatistikler

- **Mean IoU**: 0.9765 (97.65%)
- **Std IoU**: 0.0030 (0.30%)
- **Min IoU**: 0.9734 (97.34%)
- **Max IoU**: 0.9806 (98.06%)
- **Range**: 0.0072 (0.72%)

**Yorum**: 
- Fold'lar arasında çok düşük bir standart sapma görülmektedir (0.30%)
- Bu, modelin tutarlı ve güvenilir bir performans sergilediğini göstermektedir
- Tüm fold'lar %97'nin üzerinde IoU değerine ulaşmıştır

---

## 🏗️ Model Konfigürasyonu

### Hyperparameter'lar

```python
IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 3
RANDOM_SEED = 42
```

### Model Mimarisi

- **Architecture**: U-Net
- **Encoder**: EfficientNet-B4 (pretrained ImageNet)
- **Loss Function**: Combined Loss (Dice + Focal)
  - Dice Weight: 0.5
  - Focal Weight: 0.5
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR

### Data Augmentation

- Horizontal/Vertical Flip
- Random Rotation
- Shift/Scale/Rotate
- Brightness/Contrast Adjustment
- Gaussian Noise
- **Augmentation Probability**: 0.5

---

## 📁 Dosya Konumları

### Checkpoint'ler

Tüm checkpoint'ler `checkpoints/` klasöründe saklanmaktadır:

```
checkpoints/
├── unet_efficientnet-b4_fold0_best.pth  (226.48 MB)
├── unet_efficientnet-b4_fold1_best.pth  (226.48 MB)
└── unet_efficientnet-b4_fold2_best.pth (226.48 MB)
```

### TensorBoard Logları

TensorBoard logları `outputs/` klasöründe saklanmaktadır:

```
outputs/
├── unet_efficientnet-b4_fold0/
│   └── events.out.tfevents.*
├── unet_efficientnet-b4_fold1/
│   └── events.out.tfevents.*
└── unet_efficientnet-b4_fold2/
    └── events.out.tfevents.*
```

### TensorBoard Görüntüleme

TensorBoard'u açmak için:

```bash
cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2
source venv/bin/activate
tensorboard --logdir outputs
```

Sonra tarayıcıda `http://localhost:6006` adresine gidin.

---

## 🎯 Ensemble Tahmin

Final submission için 3 fold'un ensemble'ı kullanılmıştır:

- **Ensemble Yöntemi**: Weighted Average
- **Kullanılan Modeller**: 
  - unet_efficientnet-b4_fold0_best.pth
  - unet_efficientnet-b4_fold1_best.pth
  - unet_efficientnet-b4_fold2_best.pth

Ensemble tahmin, tek bir modelden daha robust ve genelleştirilmiş sonuçlar sağlamaktadır.

---

## 📈 Eğitim Süreci Analizi

### Epoch Bazlı Performans

#### Fold 0
- En iyi performans: Epoch 20 (IoU: 0.9756)
- Final performans: Epoch 30 (IoU: 0.9752)
- **Gözlem**: 20. epoch'tan sonra hafif bir overfitting görülmüştür

#### Fold 1
- En iyi performans: Epoch 29 (IoU: 0.9734)
- Final performans: Epoch 30 (IoU: 0.9734)
- **Gözlem**: Model eğitim süresince sürekli iyileşme göstermiştir

#### Fold 2
- En iyi performans: Epoch 22 (IoU: 0.9806)
- Final performans: Epoch 30 (IoU: 0.9798)
- **Gözlem**: 22. epoch'tan sonra hafif bir overfitting görülmüştür

### Genel Gözlemler

1. **Overfitting**: Fold 0 ve Fold 2'de, en iyi epoch'tan sonra hafif bir performans düşüşü görülmüştür
2. **Stabilite**: Fold 1, eğitim süresince en stabil performansı göstermiştir
3. **Erken Durdurma**: Fold 0 ve Fold 2 için early stopping kullanılabilirdi
4. **Genel Performans**: Tüm fold'lar çok yüksek IoU değerlerine ulaşmıştır (%97+)

---

## 🔍 Model Değerlendirmesi

### Güçlü Yönler

✅ **Yüksek IoU Değerleri**: Tüm fold'lar %97'nin üzerinde performans göstermiştir  
✅ **Tutarlılık**: Fold'lar arasında düşük standart sapma (0.30%)  
✅ **Robustluk**: Ensemble ile daha güvenilir tahminler  
✅ **Stabil Eğitim**: Eğitim süreci stabil ve sorunsuz ilerlemiştir  

### İyileştirme Önerileri

1. **Early Stopping**: Overfitting'i önlemek için early stopping kullanılabilir
2. **Learning Rate Scheduling**: Daha agresif learning rate scheduling denenebilir
3. **Data Augmentation**: Daha fazla augmentation tekniği eklenebilir
4. **Model Ensemble**: Farklı mimarilerin ensemble'ı denenebilir
5. **Post-processing**: Daha gelişmiş post-processing teknikleri uygulanabilir

---

## 📊 Sonuçlar ve Yorumlar

### Başarılar

- ✅ Model, %97+ IoU değerleri ile çok yüksek bir segmentasyon kalitesi elde etmiştir
- ✅ Cross-validation sonuçları, modelin genelleştirme yeteneğinin yüksek olduğunu göstermektedir
- ✅ Ensemble tahmin, daha robust sonuçlar sağlamaktadır

### Gelecek Çalışmalar

1. **Daha Fazla Epoch**: 50+ epoch ile eğitim yapılabilir
2. **5-Fold CV**: Daha robust değerlendirme için 5-fold CV kullanılabilir
3. **Farklı Mimariler**: U-Net++, DeepLabV3+ gibi farklı mimariler denenebilir
4. **Test Time Augmentation**: TTA'nın doğru implementasyonu ile performans artırılabilir
5. **Pseudo-Labeling**: Test verilerinden yararlanılabilir

---

## 📝 Notlar

- Bu rapor, checkpoint'lerden ve TensorBoard loglarından otomatik olarak oluşturulmuştur
- `cv_results.json` dosyası bulunamadı, ancak tüm bilgiler checkpoint'lerden çıkarılmıştır
- TensorBoard logları, detaylı epoch bazlı analiz için kullanılabilir
- Tüm checkpoint'ler `checkpoints/` klasöründe saklanmaktadır

---

**Rapor Oluşturulma Tarihi**: 29 Aralık 2024  
**Model Versiyonu**: U-Net + EfficientNet-B4 (3-Fold CV)  
**Toplam Eğitim Süresi**: ~3 saat (RTX 3050 Ti GPU)

---

*Bu rapor, `show_training_results.py` scripti kullanılarak oluşturulmuştur.*

