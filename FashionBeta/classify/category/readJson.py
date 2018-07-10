#-*-coding:utf-8-*-
import json
import os, sys

fashionApiPath = os.path.abspath(os.path.join(os.getcwd(),'../..'))
json_path = os.path.join(fashionApiPath,"config.json")

def readJson():
    f = file(json_path)
    s = json.load(f)
    return s

def getCategoryConfig():
    s = readJson()
    rootPath           = s["rootPath"]
    categoryModelPath  = os.path.join(rootPath, s["categoryModelPath"])
    categoryNamePath   = os.path.join(rootPath, s["categoryNamePath"])
    categorySavePath   = os.path.join(rootPath, s["categorySavePath"])

    return rootPath, categoryModelPath, categoryNamePath, categorySavePath