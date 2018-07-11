#-*-coding:utf-8-*-
from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import utils.models as models 
import utils.dataset as dataset
import json
from readJson import getAttrConfig

rootPath, attrModelPath, attrNamePath, attrSavePath = getAttrConfig()
useGpu = torch.cuda.is_available()

if useGpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def attributesNameParser(attrNamePath):
    numClasses = 0
    idx2class = {}
    for xx,line in enumerate(open(attrNamePath,'r').readlines()):
        if line!='\n':
            numClasses+=1
            idx2class[xx] = line.strip() # start from 0 
    return numClasses, idx2class



class attributesForward(object):
    def __init__(self, modelPath=attrModelPath,\
                    namePath=attrNamePath, savePath=attrSavePath, inputSize=224):
        super(attributesForward, self).__init__()
        self.modelPath = modelPath
        self.namePath  = namePath
        self.numClasses, self.idx2class = attributesNameParser(self.namePath)
        self.savePath  = savePath
        self.inputSize = inputSize
        self.model = models.sigmoid_ResnetModel('Resnet152', self.inputSize, \
                    self.numClasses, 2, 17, 5)
        self.model.load_state_dict(torch.load(modelPath), strict=True)
        if useGpu:
            self.model = self.model.cuda()
            self.model.eval()
        else:
            self.model.eval()
        
    def forward(self, jsonFile, imgPath):
        detectResult = json.load(jsonFile)
        predict_list = detectResult['bbxs']
        jsonFile.close()
        imgTensorList = dataset.getBbxImgTensor(imgPath, predict_list)
        attributesDict = {}
        attributesList = []


        for ii,imgTensor in enumerate(imgTensorList):
            input = Variable(imgTensor.cuda(),volatile=True)
            out = self.model(input)


            pred = int(out.topk(1)[1].cpu().data)
            confidence = round(float(out.topk(1)[0].cpu().data),4)
            categoryList.append([ii, pred, self.idx2class[pred], confidence])

        with open(self.savePath,"w") as jsonFile:
            categoryDict['category'] = categoryList
            json.dump(categoryDict, jsonFile)

        return categoryList, jsonFile



def relationFilter(attributesList):
    








    



