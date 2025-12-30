# 🚀 Eğitim Komutları

## Hızlı Başlangıç

### 1. Virtual Environment'i Aktif Et
```bash
cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2
source venv/bin/activate
```

### 2. Eğitimi Başlat

#### ⭐ Seçenek A: Terminal'de Görerek (ÖNERİLEN - İlk Deneme İçin)
```bash
python3 quick_start.py
```
**Bu komut eğitimi terminalde gösterir, ilerlemeyi canlı izleyebilirsiniz.**

#### Seçenek B: Arka Planda Çalıştır (Log ile)
```bash
nohup python3 quick_start.py > training.log 2>&1 &
```

#### Seçenek C: Tam Eğitim (Tüm Kombinasyonlar) - Terminal'de
```bash
python3 main.py --mode train
```

#### Seçenek D: Hem Eğitim Hem Tahmin - Terminal'de
```bash
python3 main.py --mode both
```

## Eğitim Durumunu Kontrol Et

### Canlı Log İzleme
```bash
tail -f training.log
```

### Eğitim Durumu Scripti
```bash
./check_training.sh
```

### Process Kontrolü
```bash
ps aux | grep "python3 quick_start" | grep -v grep
```

## Eğitimi Durdurma

```bash
pkill -f "python3 quick_start.py"
```

veya

```bash
pkill -f "python3 main.py"
```

## Sonuçları Görüntüleme

### Cross-Validation Sonuçları
```bash
cat outputs/cv_results.json
```

### En İyi Modelleri Listele
```bash
ls -lh checkpoints/
```

## Tahmin ve Submission Oluşturma

### Eğitim Sonrası Tahmin
```bash
python3 quick_start.py predict
```

veya

```bash
python3 main.py --mode predict
```

## Örnek Kullanım Senaryosu

### Senaryo 1: Terminal'de İzleyerek (ÖNERİLEN)
```bash
# 1. Dizine git
cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2

# 2. Virtual environment'i aktif et
source venv/bin/activate

# 3. Eğitimi başlat (terminalde göreceksiniz)
python3 quick_start.py

# Eğitimi durdurmak için: Ctrl+C
```

### Senaryo 2: Arka Planda Çalıştırma
```bash
# 1. Dizine git
cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2

# 2. Virtual environment'i aktif et
source venv/bin/activate

# 3. Eğitimi arka planda başlat
nohup python3 quick_start.py > training.log 2>&1 &

# 4. Log'u izle (Ctrl+C ile çık)
tail -f training.log

# 5. Başka bir terminalde durumu kontrol et
cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2
./check_training.sh

# 6. Eğitim tamamlandıktan sonra tahmin yap
python3 quick_start.py predict
```

## Notlar

- **CPU Kullanımı**: CPU'da eğitim uzun sürebilir (saatlerce)
- **GPU Kullanımı**: GPU varsa otomatik kullanılır (config.py'de kontrol edilir)
- **Bellek**: Batch size'ı sisteminize göre ayarlayın (config.py)
- **Log Dosyası**: `training.log` dosyasında tüm çıktılar kaydedilir

## Sorun Giderme

### Virtual Environment Bulunamadı
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CUDA Out of Memory
`config.py` dosyasında `BATCH_SIZE` değerini küçültün (örn: 4 → 2)

### Eğitim Çok Yavaş
`config.py` dosyasında:
- `NUM_EPOCHS` değerini azaltın (örn: 30 → 20)
- `N_FOLDS` değerini azaltın (örn: 3 → 2)
- Daha az kombinasyon test edin

