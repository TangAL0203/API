#-*-coding:utf-8-*-
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def ConvBNLayer(in_channels=2048, out_channels=2048, kernel_size=3, padding=1):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
    layer = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    return layer

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes=2048, planes=512, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# actually wo don't use category label, so model only has 257 outputs
Attr_Num_dict = {'category':23, 'length_of_upper_body_clothes':5, 'length_of_trousers':5, 'length_of_dresses':5, 'length_of_sleeves':8,\
'fitness_of_clothes':5, 'design_of_dresses':10, 'type_of_sleeves':10, 'type_of_trousers':7, 'type_of_dresses':12,\
'type_of_collars':10, 'type_of_waistlines':7, 'type_of_clothes_buttons':7, 'thickness_of_clothes':4, 'fabric_of_clothes':20,\
'style_of_clothes':23, 'part_details_of_clothes':72, 'graphic_elements_texture':47}

# for consumer, 17 branch multi-label
class sigmoid_ResnetModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self, arch='Resnet50', input_size=224, num_classs=257, Conv_Type=0, Branch=1, Conv_Num=1):
        super(sigmoid_ResnetModel, self).__init__()
        if arch == 'Resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif arch == 'Resnet101':
            model = torchvision.models.resnet101(pretrained=True)
        elif arch == 'Resnet152':
            model = torchvision.models.resnet152(pretrained=True)
        
        self.Conv_Type = Conv_Type
        self.Branch = Branch

        if input_size==224:
            self.avgpool_size = 7
        elif input_size==448:
            self.avgpool_size = 14
        temp = []
        for i, m in enumerate(model.children()):
            if i<=7:
                temp.append(m)
            elif i==8:
                pass
        if Branch==1:
            if Conv_Type==0:
                temp.append(nn.AvgPool2d(kernel_size=self.avgpool_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True))
            elif Conv_Type==1:
                temp+=ConvBNLayer()*Conv_Num
                temp.append(nn.AvgPool2d(kernel_size=self.avgpool_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True))
            elif Conv_Type==2:
                temp+=[Bottleneck()]*Conv_Num
                temp.append(nn.AvgPool2d(kernel_size=self.avgpool_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True))
            self.features = nn.Sequential(*temp)
            self.classifier = nn.Linear(in_features=2048, out_features=num_classs)
        elif Branch ==17:
            if Conv_Type==1:
                self.Basic_features = nn.Sequential(*temp)
                Differ_list = ConvBNLayer()*Conv_Num+[nn.AvgPool2d(kernel_size=self.avgpool_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True)]
                self.Difference_features1  = nn.Sequential(*Differ_list)
                self.Difference_features2  = nn.Sequential(*Differ_list)
                self.Difference_features3  = nn.Sequential(*Differ_list)
                self.Difference_features4  = nn.Sequential(*Differ_list)
                self.Difference_features5  = nn.Sequential(*Differ_list)
                self.Difference_features6  = nn.Sequential(*Differ_list)
                self.Difference_features7  = nn.Sequential(*Differ_list)
                self.Difference_features8  = nn.Sequential(*Differ_list)
                self.Difference_features9  = nn.Sequential(*Differ_list)
                self.Difference_features10 = nn.Sequential(*Differ_list)
                self.Difference_features11 = nn.Sequential(*Differ_list)
                self.Difference_features12 = nn.Sequential(*Differ_list)
                self.Difference_features13 = nn.Sequential(*Differ_list)
                self.Difference_features14 = nn.Sequential(*Differ_list)
                self.Difference_features15 = nn.Sequential(*Differ_list)
                self.Difference_features16 = nn.Sequential(*Differ_list)
                self.Difference_features17 = nn.Sequential(*Differ_list)
                self.classifier1  = nn.Linear(in_features=2048, out_features=5)
                self.classifier2  = nn.Linear(in_features=2048, out_features=5)
                self.classifier3  = nn.Linear(in_features=2048, out_features=5)
                self.classifier4  = nn.Linear(in_features=2048, out_features=8)
                self.classifier5  = nn.Linear(in_features=2048, out_features=5)
                self.classifier6  = nn.Linear(in_features=2048, out_features=10)
                self.classifier7  = nn.Linear(in_features=2048, out_features=10)
                self.classifier8  = nn.Linear(in_features=2048, out_features=7)
                self.classifier9  = nn.Linear(in_features=2048, out_features=12)
                self.classifier10 = nn.Linear(in_features=2048, out_features=10)
                self.classifier11 = nn.Linear(in_features=2048, out_features=7)
                self.classifier12 = nn.Linear(in_features=2048, out_features=7)
                self.classifier13 = nn.Linear(in_features=2048, out_features=4)
                self.classifier14 = nn.Linear(in_features=2048, out_features=20)
                self.classifier15 = nn.Linear(in_features=2048, out_features=23)
                self.classifier16 = nn.Linear(in_features=2048, out_features=72)
                self.classifier17 = nn.Linear(in_features=2048, out_features=47)

            elif Conv_Type==2:
                self.Basic_features = nn.Sequential(*temp)
                Differ_list = [Bottleneck()]*Conv_Num+[nn.AvgPool2d(kernel_size=self.avgpool_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True)]
                self.Difference_features1  = nn.Sequential(*Differ_list)
                self.Difference_features2  = nn.Sequential(*Differ_list)
                self.Difference_features3  = nn.Sequential(*Differ_list)
                self.Difference_features4  = nn.Sequential(*Differ_list)
                self.Difference_features5  = nn.Sequential(*Differ_list)
                self.Difference_features6  = nn.Sequential(*Differ_list)
                self.Difference_features7  = nn.Sequential(*Differ_list)
                self.Difference_features8  = nn.Sequential(*Differ_list)
                self.Difference_features9  = nn.Sequential(*Differ_list)
                self.Difference_features10 = nn.Sequential(*Differ_list)
                self.Difference_features11 = nn.Sequential(*Differ_list)
                self.Difference_features12 = nn.Sequential(*Differ_list)
                self.Difference_features13 = nn.Sequential(*Differ_list)
                self.Difference_features14 = nn.Sequential(*Differ_list)
                self.Difference_features15 = nn.Sequential(*Differ_list)
                self.Difference_features16 = nn.Sequential(*Differ_list)
                self.Difference_features17 = nn.Sequential(*Differ_list)
                self.classifier1  = nn.Linear(in_features=2048, out_features=5)
                self.classifier2  = nn.Linear(in_features=2048, out_features=5)
                self.classifier3  = nn.Linear(in_features=2048, out_features=5)
                self.classifier4  = nn.Linear(in_features=2048, out_features=8)
                self.classifier5  = nn.Linear(in_features=2048, out_features=5)
                self.classifier6  = nn.Linear(in_features=2048, out_features=10)
                self.classifier7  = nn.Linear(in_features=2048, out_features=10)
                self.classifier8  = nn.Linear(in_features=2048, out_features=7)
                self.classifier9  = nn.Linear(in_features=2048, out_features=12)
                self.classifier10 = nn.Linear(in_features=2048, out_features=10)
                self.classifier11 = nn.Linear(in_features=2048, out_features=7)
                self.classifier12 = nn.Linear(in_features=2048, out_features=7)
                self.classifier13 = nn.Linear(in_features=2048, out_features=4)
                self.classifier14 = nn.Linear(in_features=2048, out_features=20)
                self.classifier15 = nn.Linear(in_features=2048, out_features=23)
                self.classifier16 = nn.Linear(in_features=2048, out_features=72)
                self.classifier17 = nn.Linear(in_features=2048, out_features=47)

    def forward(self, x):
        if self.Branch==1:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
        elif self.Branch==17:
            x = self.Basic_features(x)
            features1  = self.Difference_features1(x)
            features2  = self.Difference_features2(x)
            features3  = self.Difference_features3(x)
            features4  = self.Difference_features4(x)
            features5  = self.Difference_features5(x)
            features6  = self.Difference_features6(x)
            features7  = self.Difference_features7(x)
            features8  = self.Difference_features8(x)
            features9  = self.Difference_features9(x)
            features10 = self.Difference_features10(x)
            features11 = self.Difference_features11(x)
            features12 = self.Difference_features12(x)
            features13 = self.Difference_features13(x)
            features14 = self.Difference_features14(x)
            features15 = self.Difference_features15(x)
            features16 = self.Difference_features16(x)
            features17 = self.Difference_features17(x)
            features1  = features1.view(features1.size(0), -1)
            features2  = features2.view(features2.size(0), -1)
            features3  = features3.view(features3.size(0), -1)
            features4  = features4.view(features4.size(0), -1)
            features5  = features5.view(features5.size(0), -1)
            features6  = features6.view(features6.size(0), -1)
            features7  = features7.view(features7.size(0), -1)
            features8  = features8.view(features8.size(0), -1)
            features9  = features9.view(features9.size(0), -1)
            features10 = features10.view(features10.size(0), -1)
            features11 = features11.view(features11.size(0), -1)
            features12 = features12.view(features12.size(0), -1)
            features13 = features13.view(features13.size(0), -1)
            features14 = features14.view(features14.size(0), -1)
            features15 = features15.view(features15.size(0), -1)
            features16 = features16.view(features16.size(0), -1)
            features17 = features17.view(features17.size(0), -1)
            return [self.classifier1(features1), self.classifier2(features2), self.classifier3(features3), self.classifier4(features4), self.classifier5(features5), \
                    self.classifier6(features6), self.classifier7(features7), self.classifier8(features8), self.classifier9(features9), self.classifier10(features10), \
                    self.classifier11(features11), self.classifier12(features12), self.classifier13(features13), self.classifier14(features14), self.classifier15(features15), \
                    self.classifier16(features16), self.classifier17(features17)]
