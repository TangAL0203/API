# -*- coding:utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend  #  default: PIL
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def getBbxImgTensor(imgPath, predict_list):
    img = default_loader(imgPath)  #  PIL.Image.Image
    imgList = []
    for predict in predict_list:
        box = (ii for ii in predict[3:])
        imgList.append(img.crop(box))
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                normalize,
                                ])
    imgTensorList = []
    for ii, im in enumerate(imgList):
        imgTensorList.append(transform(im).unsqueeze_(dim=0)) # 1x3x224x224
    del imgList

    return imgTensorList