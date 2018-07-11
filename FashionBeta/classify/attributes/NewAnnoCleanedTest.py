#-*-coding:utf-8-*-
from __future__ import print_function
import torch
import numpy as np
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
from time import time


def get_args():
    parser = argparse.ArgumentParser(description='Deepfashion Consumer-to-shop Attribute prediction using single model')

    parser.add_argument('--arch', metavar='ARCH', default='Resnet152', help='model architecture')
    parser.add_argument('--gpuId', default='1', type=str, help='GPU Id')
    parser.add_argument('--num_classs', default=257, type=int, help='attribute num_classs')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-s', '--input_size', default=224, type=int,
                        metavar='N', help='input size (default: 224)')
    parser.add_argument('--relative_path', metavar='DATA_PATH', type=str, default='../Img/',
                        help='root to relative_path', nargs=1)
    parser.add_argument('--test_path', metavar='DATA_PATH', type=str, default='../NewAnno/list_bbx_image_attr_test.txt',
                        help='root to train')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--Conv_Type', default=2, type=int, help='if add conv (default: 0)')
    parser.add_argument('--Branch', default=17, type=int, help='1 branch or 5 branch (default: 1)')
    parser.add_argument('--Conv_Num', default=5, type=int, help='depth of adding conv layers (default: 1)')

    args = parser.parse_args()
    return args

args = get_args()

print("arch           is: {}".format(args.arch))
print("gpuId          is: {}".format(args.gpuId))
print("num_classs     is: {}".format(args.num_classs))
print("batch size     is: {}".format(args.batch_size))
print("input_size     is: {}".format(args.input_size))
print("relative_path  is: {}".format(args.relative_path))
print("test_path      is: {}".format(args.test_path))
print("resume         is: {}".format(args.resume))
print("Conv_Type      is: {}".format(args.Conv_Type))
print("Branch         is: {}".format(args.Branch))
print("Conv_Num       is: {}".format(args.Conv_Num))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuId

use_gpu = torch.cuda.is_available()

Cat_Attr_recall_type = ['length_of_upper_body_clothes', 'length_of_trousers', 'length_of_dresses', 'length_of_sleeves', 'fitness_of_clothes',\
'design_of_dresses', 'type_of_sleeves', 'type_of_trousers', 'type_of_dresses', 'type_of_collars', 'type_of_waistlines', \
'type_of_clothes_buttons', 'thickness_of_clothes', 'fabric_of_clothes', 'style_of_clothes', \
'part_details_of_clothes', 'graphic_elements_texture']


Each_Attr_id = {}

lines = open('../NewAnno/Attr_id.txt').readlines()
for line in lines:
    key = line.strip().split(' ')[0]
    value = [int(i) for i in line.strip().split(' ')[1:]]
    Each_Attr_id[key] = value

def change_label(label):
    true_label = []
    for s_label in label:
        temp = [1 if i==1 else 0 for i in s_label]
        true_label.append(temp)
        
    return torch.LongTensor(true_label).cuda()  #  N1 X C


### for Branch==17

Attrs = ['length_of_upper_body_clothes', 'length_of_trousers', 'length_of_dresses', 'length_of_sleeves', 'fitness_of_clothes',\
'design_of_dresses', 'type_of_sleeves', 'type_of_trousers', 'type_of_dresses', 'type_of_collars', 'type_of_waistlines', \
'type_of_clothes_buttons', 'thickness_of_clothes', 'fabric_of_clothes', 'style_of_clothes', \
'part_details_of_clothes', 'graphic_elements_texture']

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
    batch_size = target.shape[0]
    print(output.shape)
    _, pred = output.topk(maxk, 1, True, True) # pred.shape = batch_size x maxk
    pred = pred.t() # maxk x batch_size
    correct = torch.eq(pred.cpu(), target.view(1, -1).expand_as(pred)) # target.shape = maxk x batch_size
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top1_3_prec(top1_top3, total_output, total_target):
    top1 = top1_top3[0]
    top3 = top1_top3[1]
    time1 = time()
    prec1, prec3 = accuracy(total_output, total_target, topk=(1, 3))
    # measure accuracy
    top1.update(prec1[0], total_target.shape[0])
    top3.update(prec3[0], total_target.shape[0])
    
    return top1.avg, top3.avg

def recall_precision(model, test_loader, topk=(3,5), epoch=0):
    global Cat_Attr_recall_type, Each_Attr_id, Attrs
    Attr_ids = []
    for key in Attrs:
        Attr_ids.append(Each_Attr_id[key])
    # f = open('./top3_5_recall.txt', 'w')
    # top1_pred = {}
    top3_pred = {}  #  num of samples which preded correctlt for each attr 
    top5_pred = {}
    total = {}  # num of each attr
    for i in range(257):
        top3_pred[i] = 0.0
        top5_pred[i] = 0.0
        total[i] = 0.0
    model.eval()
    time1 = time()
    total_output = []
    for batch_id, batch_label in enumerate(test_loader):
        C = batch_label[0][0].shape[0]
        H = batch_label[0][0].shape[1]
        W = batch_label[0][0].shape[2]
        batch = torch.cat((batch_label[i][0] for i in range(len(batch_label))),0).view(-1,C,H,W)  #  N C H W
        input = Variable(batch.cuda(), volatile=True)
        label = [batch_label[i][1] for i in range(len(batch_label))]
        target = change_label(label)
        cur_output = [xxx.cpu() for xxx in model(input)]
        if batch_id==0:
            total_output.append(cur_output)
            total_label = target.cpu()
        else:
            total_output.append(cur_output)
            total_label = torch.cat((total_label, target.cpu()))  # N1+N2+N3+N4... X C
    maxk = max(topk)
    tensor_total_output = []
    for output in total_output:
        for i in range(output[0].shape[0]):
            tensor_total_output.append([output[j][i].cpu().data for j in range(len(output))])

    tensor_total_label = total_label.cpu()
    # global_id = [0,156,374,554,770]
    global_id = [0, 5, 10 , 15, 23, 28, 38, 48, 55, 67, 77, 84, 91, 95, 115, 138, 210]
    multi_class_indexs = [0,1,2,6,8,9,12]
    class_branch2_prec = {0:0,1:1,2:2,6:3,8:4,9:5,12:6}
    multi_label_indexs = [3,4,5,7,10,11,13,14,15,16]
    # init preciion:
    top0_1 = AverageMeter()
    top0_3 = AverageMeter()
    top1_1 = AverageMeter()
    top1_3 = AverageMeter()
    top2_1 = AverageMeter()
    top2_3 = AverageMeter()
    top6_1 = AverageMeter()
    top6_3 = AverageMeter()
    top8_1 = AverageMeter()
    top8_3 = AverageMeter()
    top9_1 = AverageMeter()
    top9_3 = AverageMeter()
    top12_1 = AverageMeter()
    top12_3 = AverageMeter()
    top1_top3_precision = [[top0_1,top0_3],[top1_1,top1_3],[top2_1,top2_3],[top6_1,top6_3],\
                            [top8_1,top8_3],[top9_1,top9_3],[top12_1,top12_3]]
    for output, label in zip(tensor_total_output, tensor_total_label):
        # output shape [[156],[218],[180],[216],[230]]
        # label shape (1000L,)
        for i, Attr in enumerate(Cat_Attr_recall_type):
            if i in multi_label_indexs:
                some_Attr_ids = Attr_ids[i]
                some_Attr_label = label[some_Attr_ids] # only get some one branch output
                if i==12:
                    _, pred = output[i].topk(4)
                else:
                    _, pred = output[i].topk(maxk)
                Sample_id = [m for m,n in enumerate(some_Attr_label) if n==1]
                for id in Sample_id:
                    total[global_id[i]+id]+=1.0
                    if id in pred[:3]:
                        top3_pred[global_id[i]+id]+=1.0
                        top5_pred[global_id[i]+id]+=1.0
                    elif id in pred:
                        top5_pred[global_id[i]+id]+=1.0
            elif i in multi_class_indexs:
                some_Attr_ids = Attr_ids[i]
                for xxx,xxxx in enumerate(label[some_Attr_ids]):
                    if xxxx==1:
                        label_id = xxx
                        class_target = torch.LongTensor([label_id])
                        top1_3_prec(top1_top3_precision[class_branch2_prec[i]], output[i].unsqueeze_(dim=0), class_target)
                        break
                


    ff = open('./result/NewAnnoCleaned/epoch_{}_total_pre3_pre5.txt'.format(str(epoch)), 'w')
    for i in range(257):
        ff.write(str(i)+'  '+str(total[i])+'  '+str(top3_pred[i])+'  '+str(top5_pred[i])+'\n')
    ff.close()
    precision_f = open('./result/NewAnnoCleaned/Consumer_precision.txt', 'w')
    precision_f.write('Attr  top3_prec   top5prec\n')
    for ii, Attr in enumerate(Cat_Attr_recall_type):
        if ii in multi_class_indexs:
            precision_f.write(Attr+'   '+str(top1_top3_precision[class_branch2_prec[ii]][0].avg)+\
                    '   '+str(top1_top3_precision[class_branch2_prec[ii]][1].avg)+'\n')

    top3_recall = {} 
    top5_recall = {}
    overall_top3_recall = {}
    overall_top5_recall = {}
    all_top3_recall_list = []
    all_top5_recall_list = []
    all_id = []
    time1 = time()
    for i, Attr in enumerate(Cat_Attr_recall_type):
        if i<=16:
            ids = [temp_id for temp_id in Each_Attr_id[Attr]]
            print(Attr, ids)
            for hh,yy in enumerate(ids):
                ids[hh] = ids[hh]-23
            top3_recall_list = []
            top5_recall_list = []
            overall = 0.0
            c = len(ids)
            for id in ids:
                #print(id)
                if total[id]==0:
                    c-=1
                else:
                    all_id.append(id)
                    top3_recall_list.append(float(top3_pred[id]/total[id]))
                    all_top3_recall_list.append(float(top3_pred[id]/total[id]))
                    top5_recall_list.append(float(top5_pred[id]/total[id]))
                    all_top5_recall_list.append(float(top5_pred[id]/total[id]))
            if c!=0:
                top3_recall[Attr] = float(sum(top3_recall_list)/c)
                top5_recall[Attr] = float(sum(top5_recall_list)/c)
            else:
                top3_recall[Attr] = 0
                top5_recall[Attr] = 0
            try:
                overall_top3_recall[Attr] = float(sum([top3_pred[i] for i in ids])/sum([total[i] for i in ids]))
                overall_top5_recall[Attr] = float(sum([top5_pred[i] for i in ids])/sum([total[i] for i in ids]))
            except:
                overall_top3_recall[Attr] = 0
                overall_top5_recall[Attr] = 0
        else:
            top3_recall[Attr] = float(sum(all_top3_recall_list)/len(all_id))
            top5_recall[Attr] = float(sum(all_top5_recall_list)/len(all_id))
            overall_top3_recall[Attr] = float(sum([top3_pred[i] for i in all_id])/sum([total[i] for i in all_id]))
            overall_top5_recall[Attr] = float(sum([top5_pred[i] for i in all_id])/sum([total[i] for i in all_id]))

    print("overall_top3_recall is: \n")
    print(overall_top3_recall)
    print('\n')
    print("overall_top5_recall is: \n")
    print(overall_top5_recall)
    print('\n')

    return top3_recall, top5_recall, overall_top3_recall, overall_top5_recall


def main():
    time1 = time()
    global args, use_gpu
    input_size = args.input_size
    

    resume1 = './checkpoint/BBX/NewAnnoCleaned/Resnet152_64_{}_Epoch.pth'
    # resume2 = './checkpoint/BBX/sigmoid_model/Lr_0_001/Five_Branch/Bottleneck/Five_Depth/Resnet152_64_{}_Epoch.pth'
    test_loader = dataset.test_loader(args.test_path, args.relative_path, batch_size=args.batch_size, num_workers=10, pin_memory=True)
    
    model = models.sigmoid_ResnetModel(args.arch, 224, args.num_classs, 2, 17, 5)
    for i in range(49,48,-1):
        f = open('./result/NewAnnoCleaned/LR_0_001_FiveDepth_epoch_{}_top3_top5.txt'.format(str(i)), 'w')
        print("{}  epoch".format(str(i)))
        checkpoint = torch.load(resume1.format(str(i)))
        model.load_state_dict(checkpoint, strict=True)
        f.write("{} epoch\n".format(str(i)))
        if use_gpu:
            model = model.cuda()
        else:
            pass
        if args.Branch==1:
            # _, _, overall_top3_recall, overall_top5_recall = top3_5_recall(model, test_loader, topk=(3,5), epoch=i)
            pass
        elif args.Branch==17:
            _, _, overall_top3_recall, overall_top5_recall = recall_precision(model, test_loader, topk=(3,5), epoch=i)
        f.write("top-3  ")
        for key in overall_top3_recall.keys():
            f.write("{}  ".format(key))
        f.write("\n")
        for _, value in overall_top3_recall.items():
            f.write("{}  ".format(str(round(value*100, 3))))
        f.write("\n")

        f.write("top-5  ")
        for key in overall_top5_recall.keys():
            f.write("{}  ".format(key))
        f.write("\n")
        for _, value in overall_top5_recall.items():
            f.write("{}  ".format(str(round(value*100, 3))))
        f.write("\n")
        f.close()

    print("total time is: {} seconds".format(round(time()-time1, 2)))

if __name__ == "__main__":
    main()