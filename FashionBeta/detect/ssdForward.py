#-*-coding:utf-8-*-
from __future__ import print_function
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from ssd import build_ssd
from readJson import getDetectPath
import json

rootPath, detectModelPath, detectNamePath, detectSavePath, detectThresh = getDetectConfig()
useGpu = torch.cuda.is_available()

if useGpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def detectNameParser(detectNamePath):
    numClasses = 0
    idx2class = {}
    for xx,line in enumerate(open(detectNamePath,'r').readlines()):
        if line!='\n':
            numClasses+=1
            idx2class[xx] = line.strip() # start from 0 
    return numClasses+1, idx2class # 

def writeJson(imgPath, predict_list, jsfile):
    objectDict = {}
    objectDict['imgPath'] = imgPath
    objectDict['bbxs'] = predict_list
    json.dump(objectDict, jsfile)

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform(object):
    def __init__(self, size, mean=(104, 117, 123)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels

class ssdForward(object):
    """ssd inference"""
    def __init__(self, modelPath=detectModelPath,namePath=detectNamePath,\
                 savePath=detectSavePath,mode='test',size=300,useGpu=useGpu,\
                 thresh=detectThresh):
        super(ssdForward, self).__init__()
        self.modelPath = modelPath
        self.numClasses, self.idx2class = detectNameParser(namePath)
        self.savePath = savePath
        self.mode = mode
        self.size = size
        self.useGpu = useGpu
        self.thresh = thresh
        self.transform = BaseTransform(self.size, (104, 117, 123))
        self.net = build_ssd(self.mode, self.size, self.numClasses) # initialize SSD
        self.net.load_state_dict(torch.load(args.trained_model))
        self.net.eval()
        if useGpu:
            self.net = self.net.cuda()
            cudnn.benchmark = True

    def preProcess(self, imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        H, W, _ = img.shape
        x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if self.useGpu:
            x = x.cuda()
        return img, x, H, W

    def forward(self, imgPath):
        _, x, H, W = self.preProcess(imgPath)
        y = self.net(x)  # forward pass
        detections = y.cpu().data
        scale = torch.Tensor([W,H,W,H]) #  w,h,w,h
        predict_list = []
        objectId = 0
        for i in range(detections.size(1)):
            j = 0
            if i>0:
                while detections[0, i, j, 0] >= self.thresh:
                    score = round(detections[0, i, j, 0], 3)
                    labelId = i-1
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    predict_list.append([objectId,labelId,score,coords[0],coords[1],coords[2],coords[3]])
                    j += 1
                    objectId += 1
        
        with open(self.savePath,"w") as jsonFile:
            writeJson(imgPath, predict_list, jsonFile)

        return imgPath, predict_list, jsonFile