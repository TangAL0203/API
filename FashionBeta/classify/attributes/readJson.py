#-*-coding:utf-8-*-
import json
import os, sys

fashionApiPath = os.path.abspath(os.path.join(os.getcwd(),'../../'))
json_path = os.path.join(fashionApiPath,"config.json")

def readJson():
    f = file(json_path)
    s = json.load(f)
    print("read json is ok!")
    return s


def getAttrConfig():
    s = readJson()
    rootPath           = s["rootPath"]
    attrModelPath      = os.path.join(rootPath, s["attrModelPath"])
    attrNamePath       = os.path.join(rootPath, s["attrNamePath"])
    attrSavePath       = os.path.join(rootPath, s["attrSavePath"])

    return rootPath, attrModelPath, attrNamePath, attrSavePath





