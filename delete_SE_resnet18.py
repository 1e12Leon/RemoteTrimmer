from __future__ import print_function
import torch
from torchvision import datasets, transforms
from torch import nn
from SELayer import SELayer
import numpy as np

checkpoint = torch.load("model_save/resnet18_SE_train_best.pth.tar")
model = torch.load("models/resnet18_SE.pth")
model.load_state_dict(checkpoint['state_dict'])
model.to('cuda')

av = torch.load("attentions/resnet18/av_avg.pth")

model.se1 = nn.Sequential()
model.layer1[0].se1 = nn.Sequential()
model.layer1[0].se2 = nn.Sequential()
model.layer1[1].se1 = nn.Sequential()
model.layer1[1].se2 = nn.Sequential()
model.layer2[0].se1 = nn.Sequential()
model.layer2[0].se2 = nn.Sequential()
model.layer2[1].se1 = nn.Sequential()
model.layer2[1].se2 = nn.Sequential()
model.layer3[0].se1 = nn.Sequential()
model.layer3[0].se2 = nn.Sequential()
model.layer3[1].se1 = nn.Sequential()
model.layer3[1].se2 = nn.Sequential()
model.layer4[0].se1 = nn.Sequential()
model.layer4[0].se2 = nn.Sequential()
model.layer4[1].se1 = nn.Sequential()
model.layer4[1].se2 = nn.Sequential()

k1 = 0
k2 = 0
for n, m in model.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        if k not in [7,12,17]:
            m.weight.data = torch.mul(m.weight.data,av[k2])
            m.bias.data = torch.mul(m.bias.data,av[k2])
            k1+=1
            k2+=1
        else:
            k+=1

torch.save(model,"models/resnet18_delete_SE.pth")