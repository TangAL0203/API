#-*-coding:utf-8-*-
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import csv
import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path

# check if file is a image
# filename: Images/.../xxx.jpg
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

# class_to_idx: from 0 to 99
# classes: folder name
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# BBX_root = '../../Anno/BBX'  1981_Graphic_Ringer_Tee_img_00000017.jpg.txt 92 163 202 218
# Attr_root = '../../ATTR'  1981_Graphic_Ringer_Tee_img_00000017.jpg.txt   1 1 1 ... -1 -1 -1 
# BBX_Img_root = '../../Img/BBX'  1981_Graphic_Ringer_Tee/img_00000001.jpg
# Img_root = '../../Img/img'  1981_Graphic_Ringer_Tee/img_00000001.jpg

def make_dataset(txt_path, relative_path, extensions):
    images = []
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            line = line.strip()
            ImgPath = line.split(' ')[0]
            path = os.path.join(relative_path, ImgPath)
            index = [int(i) for i in line.split(' ')[1:]]  #  should keep label size same  [-1,-1,1,-1,1,-1...]
            item = (path, index)
            images.append(item)
    return images

class Deep_ConsumerDatasetFolder(data.Dataset):
    def __init__(self, txt_path, relative_path, loader, extensions, transform=None, target_transform=None):
        samples = make_dataset(txt_path, relative_path, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + txt_path + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.loader = loader
        self.extensions = extensions

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
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

class Deep_ConsumerImageFolder(Deep_ConsumerDatasetFolder):
    def __init__(self, txt_path, relative_path, transform=None, target_transform=None, 
                loader=default_loader):
        super(Deep_ConsumerImageFolder, self).__init__(txt_path, relative_path, loader, 
                                          IMG_EXTENSIONS, transform=transform, 
                                          target_transform=target_transform)

        self.imgs = self.samples



def train_loader(txt_path, relative_path, batch_size=32, num_workers=4, pin_memory=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)
    return DataLoader.DataLoader(
        Deep_ConsumerImageFolder(txt_path, relative_path,
                            transforms.Compose([
                                transforms.Resize((224,224)),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                                ])),
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = pin_memory)

def test_loader(txt_path, relative_path, batch_size=1, num_workers=4, pin_memory=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)
    return DataLoader.DataLoader(
        Deep_ConsumerImageFolder(txt_path, relative_path,
                            transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                normalize,
                                ])),
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory)