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


def get_args():
    parser = argparse.ArgumentParser(description='Deepfashion Consumer-to-shop Attribute prediction using single model')

    parser.add_argument('--arch', metavar='ARCH', default='Resnet50', help='model architecture')
    parser.add_argument('--gpuId', default='0', type=str, help='GPU Id')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--num_classs', default=257, type=int, help='attribute num_classs')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-s', '--input_size', default=224, type=int,
                        metavar='N', help='input size (default: 224)')
    parser.add_argument('--relative_path', metavar='DATA_PATH', type=str, default='../Img/',
                        help='root to relative_path', nargs=1)
    parser.add_argument('--train_path', metavar='DATA_PATH', type=str, default='../NewAnno/list_bbx_image_attr_test.txt',
                        help='root to train', nargs=1)
    parser.add_argument('--val_path', metavar='DATA_PATH', type=str, default='../NewAnno/list_bbx_image_attr_val.txt',
                        help='root to train', nargs=1)
    parser.add_argument('--test_path', metavar='DATA_PATH', type=str, default='../NewAnno/list_bbx_image_attr_test.txt',
                        help='root to train', nargs=1)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--savePath', default='./checkpoint/BBX/NewAnno', type=str, \
                        help='path to save model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--zeroTrain', default=False, action='store_true', help='choose if train from Scratch or not')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--Conv_Type', default=2, type=int, help='if add conv (default: 0)')
    parser.add_argument('--Branch', default=17, type=int, help='1 branch or 5 branch (default: 1)')
    parser.add_argument('--Conv_Num', default=5, type=int, help='depth of adding conv layers (default: 1)')



    args = parser.parse_args()
    return args

args = get_args()

print("arch           is: {}".format(args.arch))
print("gpuId          is: {}".format(args.gpuId))
print("init lr        is: {}".format(args.lr))
print("num_classs     is: {}".format(args.num_classs))
print("batch size     is: {}".format(args.batch_size))
print("input_size     is: {}".format(args.input_size))
print("relative_path  is: {}".format(args.relative_path))
print("train_path     is: {}".format(args.train_path))
print("val_path       is: {}".format(args.val_path))
print("test_path      is: {}".format(args.test_path))
print("epochs         is: {}".format(args.epochs))
print("savePath       is: {}".format(args.savePath))
print("resume         is: {}".format(args.resume))
print("momentum       is: {}".format(args.momentum))
print("zeroTrain      is: {}".format(args.zeroTrain))
print("weight_decay   is: {}".format(args.weight_decay))
print("print_freq     is: {}".format(args.print_freq))
print("Conv_Type      is: {}".format(args.Conv_Type))
print("Branch         is: {}".format(args.Branch))
print("Conv_Num       is: {}".format(args.Conv_Num))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuId

use_gpu = torch.cuda.is_available()
num_batches = 0

Each_Attr_id = {}

lines = open('../Anno/Attr_id.txt').readlines()
for line in lines:
    key = line.strip().split(' ')[0]
    value = [int(i) for i in line.strip().split(' ')[1:]]
    Each_Attr_id[key] = value


def train_val(model, train_loader, test_train_loader, val_loader, print_freq=50, optimizer=None, epoches=50, Branch=1):
    global args, num_batches
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    StepLr = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    max_train_acc = 0
    max_val_acc = 0
    if Branch==1:
        for i in range(epoches):
            if i<=50:
                StepLr.step(i)
            model.train()
            print("Epoch: ", i, "lr is: {}".format(StepLr.get_lr()))
            num_batches = sigmoid_train_epoch(model, num_batches, train_loader, print_freq=print_freq, optimizer=optimizer)
            filename = "{}_{}_{}_Epoch.pth".format(args.arch, args.batch_size, str(i))
            torch.save(model.state_dict(), osp.join(args.savePath, filename))
        print("Finished training.")
    elif Branch==17:
        for i in range(epoches):
            if i<=50:
                StepLr.step(i)
            model.train()
            print("Epoch: ", i, "lr is: {}".format(StepLr.get_lr()))
            num_batches = multi_task_train_epoch(model, Each_Attr_id, num_batches, train_loader, print_freq=print_freq, optimizer=optimizer)
            filename = "{}_{}_{}_Epoch.pth".format(args.arch, args.batch_size, str(i))
            torch.save(model.state_dict(), osp.join(args.savePath, filename))
        print("Finished training.")


def main():
    global args, num_batches, use_gpu
    if not args.zeroTrain:
        model = models.sigmoid_ResnetModel(args.arch, 224, args.num_classs, args.Conv_Type, args.Branch, args.Conv_Num)
        if args.resume:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint, strict=True)

        if use_gpu:
            model = model.cuda()
            print("Use GPU!")
        else:
            print("Use CPU!")

        train_loader = dataset.train_loader(args.train_path, args.relative_path, batch_size=1, num_workers=10, pin_memory=True)
        #test_train_loader = dataset.test_loader(args.test_path, args.relative_path, batch_size=1, num_workers=5, pin_memory=True)
        #val_loader = dataset.test_loader(args.val_path, args.relative_path, batch_size=1, num_workers=5, pin_memory=True)
        # test_loader = dataset.test_loader(rgs.test_path, args.relative_path, batch_size=1, num_workers=5, pin_memory=True)

    train_val(model, train_loader, test_train_loader=None, val_loader=None, print_freq=args.print_freq, optimizer=None, epoches=args.epochs, Branch=args.Branch)

if __name__ == "__main__":
    main()
    
