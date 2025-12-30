# 🔧 GPU Bellek Hatası Çözümü

## Sorun
RTX 3050 Ti Laptop GPU'da sadece **3.68 GB VRAM** var ve bu yetersiz.

## Yapılan Değişiklikler

### 1. Batch Size Azaltıldı
- **Önceki**: 8
- **Şimdi**: 2
- Dosya: `config.py`

### 2. Daha Hafif Encoder'lar Kullanılıyor
- **Önceki**: efficientnet-b4, efficientnet-b5 (çok büyük)
- **Şimdi**: efficientnet-b0, efficientnet-b2, resnet34 (hafif)
- Dosya: `config.py`

### 3. GPU Bellek Optimizasyonları
- Her batch'ten sonra GPU cache temizleniyor
- `non_blocking=True` kullanılıyor
- `num_workers` azaltıldı (4 → 2)
- `persistent_workers=False` eklendi
- Dosya: `train.py`

### 4. Bellek Fragmentasyonu Azaltıldı
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` eklendi
- Dosya: `train.py`

## Kullanım

### GPU Belleğini Temizleme
```bash
cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2
source venv/bin/activate
python3 clear_gpu_memory.py
```

### Eğitimi Başlatma
```bash
python3 quick_start.py
```

## Ek Öneriler

### Eğer Hala Bellek Hatası Alırsanız:

1. **Batch Size'ı 1'e düşürün** (`config.py`):
   ```python
   BATCH_SIZE = 1
   ```

2. **Daha Küçük Görüntü Boyutu** (`config.py`):
   ```python
   IMAGE_SIZE = 384  # 512 yerine
   ```

3. **Gradient Accumulation Kullanın** (train.py'de):
   - Batch size'ı küçük tutun ama gradient'leri biriktirin

4. **Mixed Precision Training** ekleyin:
   - FP16 kullanarak bellek kullanımını yarıya indirin

## Mevcut Ayarlar

- **Batch Size**: 2
- **Image Size**: 512x512
- **Encoders**: efficientnet-b0, efficientnet-b2, resnet34
- **Architectures**: unet, unetplusplus, deeplabv3plus

Bu ayarlarla RTX 3050 Ti'da çalışması gerekir! 🚀


