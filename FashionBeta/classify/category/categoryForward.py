#-*-coding:utf-8-*-
from __future__ import print_function
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from readJson import getCategoryConfig
import utils.models as models 
import json


rootPath, categoryModelPath, categoryNamePath, categorySavePath = getCategoryConfig()
useGpu = torch.cuda.is_available()

if useGpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def categoryNameParser(categoryNamePath):
    numClasses = 0
    idx2class = {}
    for xx,line in enumerate(open(categoryNamePath,'r').readlines()):
        if line!='\n':
            numClasses+=1
            idx2class[xx] = line.strip() # start from 0 
    return numClasses, idx2class

class categoryForward(object):

    def __init__(self, cv2img, imgPath, predict_list, jsonFile, modelPath=categoryModelPath,\
                 namePath=categoryNamePath,savePath=categorySavePath, inputSize=224):
        super(categoryForward, self).__init__()
        self.numClasses, self.idx2class = categoryNameParser(namePath)
        self.cv2img = cv2img
        self.H, self.W, _ = cv2img.shape
        self.imgPath = imgPath
        self.predict_list = predict_list
        self.modelPath = modelPath
        self.namePath = namePath
        self.savePath = savePath
        self.jsonFile = jsonFile
        self.inputSize = inputSize
        self.model = models.Modified_Resnet('Resnet152', self.numClasses, inputSize, False)
        self.model.load_state_dict(torch.load(modelPath), strict=True)
        if useGpu:
            self.model = self.model.cuda()
            self.model.eval()
        else:
            self.model.eval()
        
    def preProcess(self, cv2img, predict_list):
        for bbx in predict_list:
            objectId,labelId,

        return imgTensor







