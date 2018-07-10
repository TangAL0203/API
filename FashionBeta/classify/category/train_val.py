#-*-coding:utf-8-*-
from __future__ import print_function
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
    parser = argparse.ArgumentParser(description='Add layer')

    parser.add_argument('--arch', metavar='ARCH', default='Resnet50', help='model architecture')
    parser.add_argument('--gpuId', default='0', type=str, help='GPU Id')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('-s', '--input_size', default=224, type=int,
                        metavar='N', help='input size (default: 224)')
    parser.add_argument('--train_path', metavar='DATA_PATH', type=str, default='../../Img/BBX_Category/train',
                        help='root to train Image', nargs=1)
    parser.add_argument('--val_path', metavar='DATA_PATH', type=str, default='../../Img/BBX_Category/val',
                        help='root to val Image', nargs=1)
    parser.add_argument('--test_path', metavar='DATA_PATH', type=str, default='../../Img/BBX_Category/test',
                        help='root to test Image', nargs=1)

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--savePath', default='./checkpoint/BBX/zero_train', type=str, \
                        help='path to save model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--zeroTrain', default=False, action='store_true', help='choose if train from Scratch or not')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')


    args = parser.parse_args()
    return args

args = get_args()

print("arch         is: {}".format(args.arch))
print("gpuId        is: {}".format(args.gpuId))
print("init lr      is: {}".format(args.lr))
print("batch size   is: {}".format(args.batch_size))
print("input_size   is: {}".format(args.input_size))
print("train_path   is: {}".format(args.train_path))
print("test_path    is: {}".format(args.test_path))
print("val_path     is: {}".format(args.val_path))
print("epochs       is: {}".format(args.epochs))
print("savePath     is: {}".format(args.savePath))
print("resume       is: {}".format(args.resume))
print("momentum     is: {}".format(args.momentum))
print("zeroTrain    is: {}".format(args.zeroTrain))
print("weight_decay is: {}".format(args.weight_decay))
print("print_freq   is: {}".format(args.print_freq))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuId

use_gpu = torch.cuda.is_available()
num_batches = 0

def train_val(model, train_loader, test_train_loader, val_loader, test_loader, print_freq=50, optimizer=None, epoches=50):
    global args, num_batches
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    StepLr = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    max_train_acc = 0
    max_val_acc = 0
    for i in range(epoches):
        StepLr.step(i)
        model.train()
        print("Epoch: ", i, "lr is: {}".format(StepLr.get_lr()))
        num_batches = train_epoch(model, num_batches, train_loader, print_freq=print_freq, optimizer=optimizer)

        filename = "zero_{}_{}Epoch.pth".format(args.arch, str(i))
        torch.save(model.state_dict(), osp.join(args.savePath, filename))
        # try:
        #     cur_train_acc, _, cur_val_acc = get_train_val_test_acc(model, test_train_loader, val_loader, test_loader)
    
        #     if i==0:
        #         max_train_acc, max_val_acc = cur_train_acc, cur_val_acc
        #         filename = "{}_{}_{}_{}.pth".format(args.arch, 'BBX', str(cur_train_acc), str(cur_val_acc))
        #         torch.save(model.state_dict(), osp.join(args.savePath, filename))
        #     elif max_val_acc<cur_val_acc:
        #         # delete old state_dict
        #         old_filename = "{}_{}_{}_{}.pth".format(args.arch, 'BBX', str(max_train_acc), str(max_val_acc))
        #         os.remove(osp.join(args.savePath, old_filename))
        #         max_train_acc, max_val_acc = cur_train_acc, cur_val_acc
        #         filename = "{}_{}_{}_{}.pth".format(args.arch, 'BBX', str(cur_train_acc), str(cur_val_acc))
        #         torch.save(model.state_dict(), osp.join(args.savePath, filename))
        # except:
        #     print("cuda out of memory")

    print("Finished training.")

def main():
    global args, num_batches, use_gpu
    input_size = args.input_size
    if not args.zeroTrain:
        if args.resume:
            model = models.Modified_Resnet(args.arch, 46, input_size, False)
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint, strict=True)
        else:
            model = models.Modified_Resnet(args.arch, 46, input_size, not args.zeroTrain)
    elif args.zeroTrain:
        model = models.Modified_Resnet(args.arch, 46, input_size, not args.zeroTrain)

    if use_gpu:
        model = model.cuda()
        print("Use GPU!")
    else:
        print("Use CPU!")

    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    train_root = args.train_path
    val_root = args.val_path
    test_root = args.test_path

    train_loader = dataset.train_loader(train_root, args.input_size, batch_size=args.batch_size, num_workers=5, pin_memory=True)
        # test_train_loader = dataset.test_loader(train_root, args.input_size, batch_size=1, num_workers=5, pin_memory=True)
        # val_loader = dataset.test_loader(val_root, args.input_size, batch_size=1, num_workers=5, pin_memory=True)
        # test_loader = dataset.test_loader(test_root, args.input_size, batch_size=1, num_workers=5, pin_memory=True)
    train_val(model, train_loader, test_train_loader=None, val_loader=None, test_loader=None, print_freq=args.print_freq, optimizer=None, epoches=args.epochs)

if __name__ == "__main__":
    main()