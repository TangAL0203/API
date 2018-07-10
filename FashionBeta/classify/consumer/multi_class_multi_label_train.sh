#!/usr/bin/env sh
echo "Deepfashion Consumer-to-shop Attribute prediction using single model"
python multi_class_multi_label_train.py --arch Resnet152 --batch_size 64 --epochs 50 --gpuId 1 --momentum 0.9 --weight_decay 5e-4 --print_freq 50 --savePath ./checkpoint/BBX/NewAnnoCleaned --Conv_Type 2 --Branch 17 --Conv_Num 5