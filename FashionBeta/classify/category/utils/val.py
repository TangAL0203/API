#-*-coding:utf-8-*-
import os
import os.path as osp
import numpy as np
import shutil
import csv

'''
Evaluation Criteria
Refrece link: https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.80313a26qpAGkl&raceId=231649
'''

'''
prediction csv file: 
ImageName   AttrKey              AttrValueProbs
xxx.jpg     pant_length_labels   0.07,0.16,0.22,0.27,0.25,0.074
'''

with open('first_test.csv', 'wb') as csvfile:
    fieldnames = ['imgname', 'imgclass']
    writer = csv.DictWriter(csvfile, delimiter=' ', fieldnames=fieldnames) # 写入列的时候，以' '空格作为分割符号(在windows下用)
    # writer.writeheader() # 写入头信息
    with open('./first_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            name, classId = line.split(' ')
            writer.writerow({'imgname': name, 'imgclass': classId})


