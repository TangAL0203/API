# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def train_loader(path, input_size=224, batch_size=32, num_workers=4, pin_memory=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)
    if input_size==448:
        ResizeSize = 480
    elif input_size==224:
        ResizeSize = 256
    return data.DataLoader(
        datasets.ImageFolder(path,
                            transforms.Compose([
                                transforms.Resize(ResizeSize),
                                transforms.RandomCrop(input_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                                ])),
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = pin_memory)

def test_loader(path, input_size=224, batch_size=1, num_workers=4, pin_memory=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)
    if input_size==448:
        ResizeSize = 480
    elif input_size==224:
        ResizeSize = 256
    return data.DataLoader(
        datasets.ImageFolder(path,
                            transforms.Compose([
                                transforms.Resize(ResizeSize),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                normalize,
                                ])),
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


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
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def getBbxImgTensor(imgPath, predict_list):
    img = default_loader(path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                                ])


    img = transform(img)
    
    img = img.unsqueeze(dim=0)
    for target in sorted(os.listdir(root)):
        if target in path:
            label = class_to_idx[target]

    return (img, label)