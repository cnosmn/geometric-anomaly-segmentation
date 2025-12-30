# 🕵️‍♂️ Geometrik Anomali Segmentasyonu

Bu proje, görüntülerdeki geometrik anomalileri (istila birimlerini) tespit etmek için geliştirilmiş profesyonel bir deep learning çözümüdür.

## 📋 Görev Özeti

- **Hedef**: Görüntülerdeki geometrik anomalileri (noise, invert, solid color, şekiller: yıldızlar, daireler, dikdörtgenler, çokgenler) binary maskeler olarak segmentasyon
- **Metrik**: Mean IoU (Intersection over Union)
- **Veri**: 400 eğitim görüntüsü + maskeleri, 200 test görüntüsü
- **Format**: 512x512 RGB görüntüler, binary maskeler
- **Çıktı**: RLE (Run-Length Encoding) formatında CSV submission dosyası

## 🚀 Kurulum

### 1. Gereksinimler

```bash
cd iaio-problem-2
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Veri Yapısı

Proje dizini şu şekilde organize edilmelidir:

```
iaio-problem-2/
├── train/
│   ├── images/     # 400 eğitim görüntüsü
│   └── masks/      # 400 eğitim maskesi
├── test/
│   └── images/     # 200 test görüntüsü
└── ...
```

## 📁 Proje Yapısı

```
iaio-problem-2/
├── config.py              # Tüm hyperparameter'lar ve ayarlar
├── dataset.py             # PyTorch Dataset sınıfları (normal + TTA)
├── models.py              # Model mimarileri ve loss fonksiyonları
├── train.py               # Cross-validation ile eğitim
├── inference.py           # Test seti tahmin ve submission
├── utils.py               # Yardımcı fonksiyonlar (RLE, IoU, post-processing)
├── main.py               # Ana pipeline scripti
├── test_models.py         # Mevcut modelleri test etme
├── create_submission_fixed.py  # Düzeltilmiş submission oluşturma
├── visualize_predictions.py    # Tahmin görselleştirme
├── visualize_submission.py     # Submission görselleştirme
├── analyze_data.py        # Veri analizi scripti
├── requirements.txt       # Python bağımlılıkları
├── KOMUTLAR.md          # Hızlı komut referansı
└── README.md            # Bu dosya
```

## 🎯 Kullanım

### 1. Veri Analizi

```bash
python analyze_data.py
```

Bu script veri setini analiz eder:
- Görüntü boyutları
- Maske tipleri
- Anomali piksel sayıları
- Kapsama istatistikleri

### 2. Eğitim (Cross-Validation ile Model Karşılaştırma)

```bash
python main.py --mode train
```

Bu komut:
- Farklı model mimarilerini (U-Net, U-Net++, DeepLabV3+) test eder
- Farklı encoder'ları (EfficientNet, ResNet) test eder
- 3-fold cross-validation yapar (config'de `N_FOLDS=3`)
- En iyi modelleri `checkpoints/` klasörüne kaydeder
- Sonuçları `outputs/cv_results.json` dosyasına yazar

### 3. Tahmin ve Submission Oluşturma

#### Yöntem 1: Düzeltilmiş Submission (Önerilen)

```bash
python create_submission_fixed.py
```

Bu script:
- TTA kullanmaz (daha doğru sonuçlar için)
- `visualize_predictions.py` ile aynı mantığı kullanır
- Ensemble tahmin yapar (3 fold modeli)
- `submission.csv` dosyası oluşturur

#### Yöntem 2: Standart Inference

```bash
python main.py --mode predict
```

veya

```bash
python test_models.py
```

### 4. Görselleştirme

#### Tahmin Görselleştirmesi

```bash
python visualize_predictions.py --num_images 10
```

Test görüntülerinde tahmin edilen maskeleri görselleştirir.

#### Submission Görselleştirmesi

```bash
python visualize_submission.py --html
```

Submission.csv'deki her görüntü için orijinal + maske görselleştirmesi oluşturur ve HTML viewer hazırlar.

HTML viewer'ı açmak için:
```bash
xdg-open submission_viewer.html
```

## 🏗️ Kullanılan Model Mimarileri

### 1. U-Net (Seçilen Model)
- **Encoder**: EfficientNet-B4
- **Neden Seçildi**: Cross-validation sonuçlarına göre en iyi performans
- **Özellikler**:
  - Encoder-decoder yapısı
  - Skip connections ile detay korunması
  - Segmentasyon görevleri için optimize edilmiş

### 2. Test Edilen Diğer Mimariler
- **U-Net++**: Geliştirilmiş U-Net versiyonu (nested skip connections)
- **DeepLabV3+**: Atrous convolution ile daha iyi context
- **FPN**: Feature Pyramid Network (test aşamasında kapatıldı)

## 🔧 Kullanılan Encoder'lar

### EfficientNet-B4 (Final Model)
- **Neden Seçildi**: En iyi performans/doğruluk dengesi
- **Özellikler**: 
  - Compound scaling ile optimize edilmiş
  - Güçlü feature extraction
  - Orta seviye model boyutu

### Test Edilen Diğer Encoder'lar
- **EfficientNet-B0**: En hafif (küçük GPU için)
- **EfficientNet-B2**: Orta seviye
- **ResNet-34**: Hafif alternatif
- **EfficientNet-B5**: Çok büyük (GPU bellek sınırlaması nedeniyle kullanılamadı)

## 📊 Loss Fonksiyonları

### Combined Loss (Dice + Focal)
- **Dice Loss** (Ağırlık: 0.5): Küçük objeler için optimize, class imbalance'a karşı dirençli
- **Focal Loss** (Ağırlık: 0.5): Zor örnekleri vurgular, kolay örnekleri down-weight eder
- **Neden**: Binary segmentation görevinde en etkili kombinasyon

## 🎨 Data Augmentation

Eğitim sırasında kullanılan augmentation'lar:

- **Horizontal/Vertical Flip**: Yatay ve dikey çevirme
- **Random Rotation**: Rastgele döndürme
- **Shift/Scale/Rotate**: Kombine geometrik transformasyonlar
- **Brightness/Contrast Adjustment**: Parlaklık ve kontrast ayarları
- **Gaussian Noise**: Gürültü ekleme (robustluk için)

**Augmentation Probability**: 0.5 (her augmentasyon %50 şansla uygulanır)

## 🔄 Test Time Augmentation (TTA) - Kullanılmadı

**Önemli Not**: İlk submission'larda TTA kullanıldı ancak transform'ların geri çevrilmemesi nedeniyle kötü sonuçlar verdi. Final submission'da TTA kapatıldı.

### TTA Sorunu ve Çözümü

**Sorun**:
- TTA transform'ları (flip, rotate) uygulanıyordu
- Ancak tahminler geri çevrilmeden birleştiriliyordu
- Bu, maskelerin yanlış yerleşmesine yol açıyordu

**Çözüm**:
- `create_submission_fixed.py` scripti oluşturuldu
- TTA kullanılmadan doğrudan tahmin yapılıyor
- `visualize_predictions.py` ile aynı mantık kullanılıyor

## 🎯 Ensemble

Final submission'da **3-fold ensemble** kullanıldı:
- Her fold için ayrı model eğitildi
- Tahminler ortalama alınarak birleştirildi
- Daha robust ve genelleştirilmiş sonuçlar elde edildi

## 🔧 Post-Processing

### Morphological Operations
- **Opening**: Küçük gürültüleri temizler
- **Closing**: Küçük delikleri doldurur
- **Kernel Size**: 3x3

### Threshold Ayarları
- **RLE_THRESHOLD**: 0.3 (düşürüldü - daha hassas tespit için)
- Binary threshold olarak kullanılır

## 📤 RLE Encoding

Görev dokümanına göre implement edildi:

- **Column-major flattening**: Yukarıdan aşağıya, sütun sütun tarama
- **1-based indexing**: Pikseller 1'den başlayarak numaralandırılır
- **Run-length encoding**: Başlangıç pikseli ve uzunluk çiftleri

Örnek:
```
Maske:  [0, 0, 1, 0]
        [1, 0, 1, 0]
        [1, 0, 1, 0]
        [0, 0, 0, 0]

RLE: "2 2 9 3"
```

## 🚀 Yapılan İyileştirmeler ve Optimizasyonlar

### 1. GPU Bellek Optimizasyonu

**Sorun**: RTX 3050 Ti (3.68GB VRAM) için CUDA out of memory hataları

**Çözümler**:
- `BATCH_SIZE`: 8 → 2'ye düşürüldü
- Hafif encoder'lar kullanıldı (EfficientNet-B0, B2, ResNet-34)
- `num_workers`: 4 → 2'ye düşürüldü
- `torch.cuda.empty_cache()` eklendi
- `non_blocking=True` ile asenkron transfer

### 2. Model Seçimi

**Süreç**:
1. Başlangıçta tüm mimariler test edildi
2. Cross-validation sonuçlarına göre U-Net + EfficientNet-B4 seçildi
3. 3-fold CV ile eğitim tamamlandı

**Sonuç**: U-Net + EfficientNet-B4 en iyi performansı gösterdi

### 3. TTA Sorunu ve Düzeltme

**Sorun**: 
- İlk submission'larda TTA kullanıldı
- Transform'lar geri çevrilmedi
- Kötü segmentasyon sonuçları

**Çözüm**:
- `create_submission_fixed.py` oluşturuldu
- TTA kapatıldı
- `visualize_predictions.py` ile aynı mantık kullanıldı
- Daha doğru sonuçlar elde edildi

### 4. Threshold Optimizasyonu

**Süreç**:
- Başlangıç: `RLE_THRESHOLD = 0.5`
- Optimizasyon: `RLE_THRESHOLD = 0.3` (daha hassas tespit)
- Fallback mekanizması eklendi (boş maskeler için)

### 5. Post-Processing İyileştirmeleri

- Morphological operations eklendi
- Küçük gürültüler temizlendi
- Maske kalitesi artırıldı

### 6. Görselleştirme Araçları

**Oluşturulan Scripts**:
- `visualize_predictions.py`: Test görüntülerinde tahmin görselleştirme
- `visualize_submission.py`: Submission.csv'den görselleştirme oluşturma
- HTML viewer: Tüm görüntüleri tek sayfada görüntüleme

## 📈 Eğitim Detayları

### Hyperparameter'lar

```python
IMAGE_SIZE = 512
BATCH_SIZE = 2  # GPU bellek sınırlaması nedeniyle
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 3
```

### Eğitim Süreci

1. **Cross-Validation**: 3-fold CV ile model performansı değerlendirildi
2. **Early Stopping**: Validation loss'a göre en iyi model kaydedildi
3. **Model Checkpointing**: Her fold için en iyi model `checkpoints/` klasörüne kaydedildi

### Final Model

- **Mimari**: U-Net
- **Encoder**: EfficientNet-B4
- **Folds**: 3 (fold0, fold1, fold2)
- **Ensemble**: 3 modelin ortalaması

## 📊 Sonuçlar ve İstatistikler

### Submission İstatistikleri

- **Toplam Görüntü**: 200
- **Anomali Tespit Edilen**: 200
- **Anomali Tespit Edilmeyen**: 0

### Model Performansı

Cross-validation sonuçları `outputs/cv_results.json` dosyasında saklanır:
- Her model için mean IoU ve std
- Her fold için IoU değerleri
- En iyi model otomatik seçilir

## 💡 Kullanım İpuçları

### 1. GPU Kullanımı
- CUDA varsa otomatik kullanılır, yoksa CPU'ya geçer
- GPU bellek sınırlaması için `BATCH_SIZE` ayarlanabilir

### 2. Bellek Yönetimi
- Küçük GPU'lar için `BATCH_SIZE=2` önerilir
- Daha büyük GPU'lar için `BATCH_SIZE=4` veya `8` kullanılabilir
- `IMAGE_SIZE` azaltılabilir (512 → 384) bellek için

### 3. Eğitim Süresi
- Cross-validation tüm kombinasyonları test eder, zaman alabilir
- Hızlı test için `N_FOLDS=3` ve `NUM_EPOCHS=30` kullanıldı
- Tam eğitim için `N_FOLDS=5` ve `NUM_EPOCHS=50+` önerilir

### 4. Model Seçimi
- Sonuçlara göre en iyi kombinasyonu seçin
- Sadece en iyi modeli eğitin (zaman tasarrufu için)

## 🐛 Sorun Giderme

### CUDA Out of Memory
```bash
# config.py'de BATCH_SIZE'ı küçültün
BATCH_SIZE = 1  # veya 2

# Daha hafif encoder kullanın
ENCODERS = ['efficientnet-b0', 'resnet34']
```

### Yavaş Eğitim
```bash
# NUM_EPOCHS'ı azaltın
NUM_EPOCHS = 20

# Daha az model test edin
ARCHITECTURES = ['unet']  # Sadece U-Net
```

### Düşük IoU
- Data augmentation'ı artırın
- Daha fazla epoch eğitin
- Learning rate'i ayarlayın
- Farklı loss kombinasyonları deneyin

### Submission Hataları
- RLE encoding'i kontrol edin (`test_rle_simple.py`)
- Threshold değerini ayarlayın
- Post-processing'i kontrol edin

## 📝 Önemli Dosyalar

- **Checkpoint'ler**: `checkpoints/` klasörüne kaydedilir
- **TensorBoard Logları**: `outputs/` altında
- **Submission Dosyası**: `submission.csv`
- **Görselleştirmeler**: 
  - `visualizations/`: Tahmin görselleştirmeleri
  - `submission_visualizations/`: Submission görselleştirmeleri
- **HTML Viewer**: `submission_viewer.html`

## 🔍 Hızlı Komut Referansı

Detaylı komutlar için `KOMUTLAR.md` dosyasına bakın.

### Eğitim
```bash
python main.py --mode train
```

### Tahmin
```bash
python create_submission_fixed.py
```

### Görselleştirme
```bash
python visualize_submission.py --html
xdg-open submission_viewer.html
```

### Model Test
```bash
python test_models.py
```

## 🎓 Teknik Detaylar

### Model Mimarisi Detayları

**U-Net**:
- Encoder: EfficientNet-B4 (pretrained ImageNet)
- Decoder: U-Net decoder with skip connections
- Output: Single channel binary mask

**Loss Function**:
```python
loss = 0.5 * DiceLoss() + 0.5 * FocalLoss()
```

**Optimizer**:
- Adam optimizer
- Learning rate: 1e-4
- Weight decay: 1e-4

### Data Pipeline

1. **Loading**: PIL Image → NumPy array
2. **Augmentation**: Albumentations transforms
3. **Normalization**: ImageNet mean/std
4. **Tensor Conversion**: NumPy → PyTorch tensor

### Inference Pipeline

1. **Model Loading**: Checkpoint'ten model yükleme
2. **Preprocessing**: Normalization ve resize
3. **Prediction**: Model inference
4. **Post-processing**: Threshold ve morphological operations
5. **RLE Encoding**: Binary mask → RLE string
6. **CSV Generation**: Submission dosyası oluşturma

## 📚 Referanslar

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)
- [PyTorch](https://pytorch.org/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)

## 🎯 Gelecek İyileştirmeler

1. **Daha Fazla Epoch**: Tam eğitim için 50+ epoch
2. **5-Fold CV**: Daha robust değerlendirme
3. **Learning Rate Scheduling**: Cosine annealing veya reduce on plateau
4. **Pseudo-Labeling**: Test verilerinden yararlanma
5. **Model Ensemble**: Farklı mimarilerin kombinasyonu
6. **Advanced Augmentation**: MixUp, CutMix gibi teknikler
7. **Attention Mechanisms**: Attention U-Net gibi gelişmiş mimariler

## 📄 Lisans

Bu proje açık kaynak olarak paylaşılmıştır.

---

**İyi şanslar! Pikselleri duyuyor musun? Yaklaşıyorlar...** 🚀

**Not**: Bu README, projenin final durumunu yansıtmaktadır. Tüm iyileştirmeler ve optimizasyonlar dokümante edilmiştir.
