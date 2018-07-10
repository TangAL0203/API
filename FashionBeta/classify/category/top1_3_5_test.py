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
from time import time
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser(description='Test Deepfashion category classification and attribute benchmark')

    parser.add_argument('--arch', metavar='ARCH', default='Resnet152', help='model architecture')
    parser.add_argument('--gpuId', default='2', type=str, help='GPU Id')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-s', '--input_size', default=224, type=int,
                        metavar='N', help='input size (default: 224)')
    parser.add_argument('--test_path', metavar='DATA_PATH', type=str, default='../../Img/BBX_Category/test',
                        help='root to test Image', nargs=1)
    parser.add_argument('--resume', type=str, default = './checkpoint/BBX/zero_train/zero_Resnet152_49Epoch.pth', 
                        metavar='PATH', help='path to latest checkpoint')

    args = parser.parse_args()
    return args

args = get_args()

print("arch         is: {}".format(args.arch))
print("gpuId        is: {}".format(args.gpuId))
print("batch size   is: {}".format(args.batch_size))
print("input_size   is: {}".format(args.input_size))
print("test_path    is: {}".format(args.test_path))
print("resume       is: {}".format(args.resume))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuId

use_gpu = torch.cuda.is_available()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)  #  if topk=(1,5) maxk=5
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # pred.shape = batch_size x maxk
    pred = pred.t() # maxk x batch_size
    correct = torch.eq(pred.cpu().data, target.view(1, -1).expand_as(pred)) # target.shape = maxk x batch_size
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top1_3_5(f, epoch, model, test_loader):
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    
    time1 = time()
    for i, (input, target) in enumerate(test_loader):
        output = model(Variable(input.cuda(), volatile=True))
        prec1, prec3, prec5 = accuracy(output, target, topk=(1, 3, 5))
        # measure accuracy
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))
        top5.update(prec5[0], input.size(0))
    f.write("{} epoch".format(str(epoch)))
    f.write("top1 acc : {}\n".format(str(round(top1.avg, 2))))
    f.write("top3 acc : {}\n".format(str(round(top3.avg, 2))))
    f.write("top5 acc : {}\n".format(str(round(top5.avg, 2))))

    print("top1 acc : {}".format(str(round(top1.avg, 2))))
    print("top3 acc : {}".format(str(round(top3.avg, 2))))
    print("top5 acc : {}".format(str(round(top5.avg, 2))))
    
    print("eval cost total time: {} seconds".format(round(time()-time1, 1)))
    return top1.avg, top3.avg, top5.avg


def main():
    global args, use_gpu
    input_size = args.input_size
    model = models.Modified_Resnet(args.arch, 46, input_size)
    if use_gpu:
        model = model.cuda()
        print("Use GPU!")
    else:
        print("Use CPU!")
    test_root = args.test_path
    test_loader = dataset.test_loader(test_root, args.input_size, batch_size=args.batch_size, num_workers=10, pin_memory=True)
    f = open('./top1_3_5.txt', 'w')
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint, strict=True)
    top1_3_5(f,49 , model, test_loader)
    f.close()

    
    

if __name__ == "__main__":
    main()
