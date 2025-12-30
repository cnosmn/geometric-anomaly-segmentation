#!/bin/bash
# Eğitim scripti

cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2
source venv/bin/activate

echo "=========================================="
echo "Eğitim başlatılıyor..."
echo "=========================================="
echo ""

python3 quick_start.py 2>&1 | tee training.log

echo ""
echo "=========================================="
echo "Eğitim tamamlandı!"
echo "Log dosyası: training.log"
echo "=========================================="


