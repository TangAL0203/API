#! /bin/bash
echo "Deepfashion Consumer-to-shop Attribute prediction using single model"
python NewAnnoCleanedTest.py --arch Resnet152 --batch_size 16 --gpuId 3 