#!/usr/bin/env sh
echo "Deepfashion Consumer-to-shop Attribute prediction using single model"
python new_top_3_5_recall.py --arch Resnet152 --batch_size 16 --gpuId 1 