#-*-coding:utf-8-*-
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# import pretrainedmodels  # for pytorch v0.4 python 3.5

class Modified_Resnet(nn.Module):
    """docstring for ClassName"""
    def __init__(self, arch='Resnet152', num_classs=46, input_size=224, pretrained=True):
        super(Modified_Resnet, self).__init__()
        if arch=='Resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif arch=='Resnet101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif arch=='Resnet152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        self.num_classs = num_classs
        if input_size==224:
            self.avgpool_size = 7
        elif input_size==448:
            self.avgpool_size = 14
        temp = []
        for i, m in enumerate(model.children()):
            if i<=7:
                temp.append(m)
            elif i==8:
                temp.append(nn.AvgPool2d(kernel_size=self.avgpool_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True))
            else:
                self.classifier = nn.Linear(in_features=2048, out_features=num_classs)
        self.features = nn.Sequential(*temp)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x