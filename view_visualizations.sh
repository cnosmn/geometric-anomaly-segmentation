#!/bin/bash
# Görselleştirmeleri görüntüle

cd /home/cnosmn/Desktop/ahmet_tunc_uge/iaio-problem-2

echo "=========================================="
echo "Görselleştirmeler"
echo "=========================================="
echo ""

if [ -d "visualizations" ]; then
    echo "Oluşturulan görselleştirmeler:"
    ls -1 visualizations/*.png | head -10
    echo ""
    echo "Toplam: $(ls visualizations/*.png | wc -l) dosya"
    echo ""
    echo "Görüntülemek için:"
    echo "  - Linux: xdg-open visualizations/prediction_img_0000.png"
    echo "  - Veya dosya yöneticisinde visualizations/ klasörünü açın"
else
    echo "Görselleştirme klasörü bulunamadı!"
    echo "Önce visualize_predictions.py çalıştırın."
fi


