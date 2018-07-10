#!/usr/bin/env sh
echo "Deepfashion Consumer-to-shop Attribute prediction using single model"
python single_model_train_val.py --arch Resnet152 --batch_size 64 --epochs 50 --gpuId 0 --momentum 0.9 --weight_decay 5e-4 --print_freq 50 --savePath ./checkpoint/BBX/NewAnno --Conv_Type 2 --Branch 17 --Conv_Num 5