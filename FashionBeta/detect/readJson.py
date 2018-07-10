#-*-coding:utf-8-*-
import json
import os, sys

fashionApiPath = os.path.abspath(os.path.join(os.getcwd(),'..'))
json_path = os.path.join(fashionApiPath,"config.json")

def readJson():
    f = file(json_path)
    s = json.load(f)
    print("read json is ok!")
    return s
    
def getDetectConfig():
    s = readJson()
    rootPath           = s["rootPath"]
    detectModelPath    = os.path.join(rootPath, s["detectModelPath"])
    detectNamePath     = os.path.join(rootPath, s["detectNamePath"])
    detectSavePath     = os.path.join(rootPath, s["detectSavePath"])
    detectThresh       = s["rootPath"]

    return rootPath, detectModelPath, detectNamePath, detectSavePath, detectThresh

def getCategoryConfig():
    s = readJson()
    rootPath           = s["rootPath"]
    categoryModelPath  = os.path.join(rootPath, s["categoryModelPath"])
    categoryNamePath   = os.path.join(rootPath, s["categoryNamePath"])

    return rootPath, categoryModelPath, categoryNamePath


def getAttrConfig():
    s = readJson()
    rootPath           = s["rootPath"]
    attrModelPath      = os.path.join(rootPath, s["attrModelPath"])
    attrNamePath       = os.path.join(rootPath, s["attrNamePath"])

    return rootPath, attrModelPath, attrNamePath





