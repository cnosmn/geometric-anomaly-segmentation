#!/bin/bash
# Eğitim durumunu kontrol et

cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2

echo "=========================================="
echo "Eğitim Durumu"
echo "=========================================="
echo ""

# Process kontrolü
if pgrep -f "python3 quick_start.py" > /dev/null; then
    echo "✅ Eğitim ÇALIŞIYOR"
    echo ""
    ps aux | grep "python3 quick_start" | grep -v grep
else
    echo "❌ Eğitim DURMUŞ"
fi

echo ""
echo "=========================================="
echo "Son Log Satırları (son 20 satır)"
echo "=========================================="
if [ -f training.log ]; then
    tail -20 training.log
else
    echo "Log dosyası henüz oluşturulmadı"
fi

echo ""
echo "=========================================="
echo "Checkpoint'ler"
echo "=========================================="
if [ -d checkpoints ]; then
    ls -lh checkpoints/ | tail -10
    echo ""
    echo "Toplam checkpoint sayısı: $(ls checkpoints/*.pth 2>/dev/null | wc -l)"
else
    echo "Checkpoint klasörü henüz oluşturulmadı"
fi

echo ""
echo "=========================================="
echo "Sonuçlar"
echo "=========================================="
if [ -f outputs/cv_results.json ]; then
    echo "✅ Sonuçlar dosyası mevcut"
    echo "En iyi modeller:"
    python3 -c "import json; data=json.load(open('outputs/cv_results.json')); sorted_data=sorted(data.items(), key=lambda x: x[1]['mean_iou'], reverse=True)[:3]; [print(f\"  {k}: IoU={v['mean_iou']:.4f}\") for k,v in sorted_data]" 2>/dev/null || echo "  (JSON parse edilemedi)"
else
    echo "Sonuçlar dosyası henüz oluşturulmadı"
fi

echo ""


