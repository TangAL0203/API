#-*-coding:utf-8-*-
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import utils.models as models
from utils.train_Components import *
import torchvision
import torchvision.transforms as transforms
import shutil
import math
import os
import os.path as osp
import math
import argparse
from collections import OrderedDict
import csv


def get_args():
    parser = argparse.ArgumentParser(description='Deepfashion Consumer-to-shop Attribute prediction using single model')
    parser.add_argument('--gpuId', default='0', type=str, help='GPU Id')
    parser.add_argument('--num_classs', default=257, type=int, help='attribute num_classs')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-s', '--input_size', default=224, type=int,
                        metavar='N', help='input size (default: 224)')
    parser.add_argument('--relative_path', metavar='DATA_PATH', type=str, default='../Img/',
                        help='root to relative_path', nargs=1)
    parser.add_argument('--data_path', metavar='DATA_PATH', type=str, default='../NewAnnoCleaned/list_bbx_image_attr_all.txt',
                        help='root to train', nargs=1)
    parser.add_argument('--resume', default='./checkpoint/BBX/NewAnnoCleaned/Resnet152_64_49_Epoch.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--Branch', default=17, type=int, help='1 branch or 5 branch (default: 1)')
    

    args = parser.parse_args()
    return args

# 200_list_bbx_image_attr_all.txt
# list_bbx_image_attr_all.txt
args = get_args()

print("gpuId            is: {}".format(args.gpuId))
print("num_classs       is: {}".format(args.num_classs))
print("batch size       is: {}".format(args.batch_size))
print("input_size       is: {}".format(args.input_size))
print("relative_path    is: {}".format(args.relative_path))
print("data_path        is: {}".format(args.data_path))
print("resume           is: {}".format(args.resume))
print("Branch           is: {}".format(args.Branch))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuId

use_gpu = torch.cuda.is_available()
num_batches = 0

Each_Attr_id = {}

lines = open('../NewAnnoCleaned/Attr_id.txt').readlines()
for line in lines:
    key = line.strip().split(' ')[0]
    value = [int(i) for i in line.strip().split(' ')[1:]]
    Each_Attr_id[key] = value


def get_data_loss(txtFile, csvfile, writer, model, data_loader):
    global args, Each_Attr_id
    print("Start testing")
    if args.Branch==1:
        pass
    elif args.Branch==17:
        mix_multi_task_test_loss_epoch(txtFile, csvfile, writer, model, Each_Attr_id, data_loader)
        
        
def main():
    global args, num_batches, use_gpu
    model = models.sigmoid_ResnetModel('Resnet152', 224, args.num_classs, 2, 17, 5)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=True)
        
    if use_gpu:
        model = model.cuda()
        print("Use GPU!")
    else:
        print("Use CPU!")
    
    csvfile = open('./Consumer_loss.csv', 'wb')
    fieldnames = ['img path', 'total loss']
    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames) #
    writer.writeheader()
    data_loader = dataset.train_loader(args.data_path, args.relative_path, batch_size=args.batch_size, num_workers=10, pin_memory=True)
    txtFile = open('./Consumer_confidence.txt','w')
    get_data_loss(txtFile, csvfile, writer, model, data_loader)

if __name__ == "__main__":
    main()
    