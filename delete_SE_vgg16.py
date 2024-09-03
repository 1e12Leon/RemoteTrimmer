from __future__ import print_function
import torch
from torchvision import datasets, transforms
from torch import nn
from SELayer import SELayer
import numpy as np

checkpoint = torch.load("model_save/vgg16_SE_train_best.pth.tar")
checkpoint2 = torch.load("model_save/vgg16_bn_train_best.pth.tar")
model = torch.load("models/vgg16/vgg16_SE.pth")
model.load_state_dict(checkpoint['state_dict'])
model.to('cuda')
av = torch.load("attentions/vgg16/av_avg.pth")

#----------------delete SELayer-------------
list = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]
for i in list:
    model.features[i] = nn.Sequential()

#----------------update BN(w,b)-------------
k = 0
for n, m in model.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data = torch.mul(m.weight.data,av[k])
        m.bias.data = torch.mul(m.bias.data,av[k])
        k += 1

torch.save(model,"models/vgg16_SE_updateBN.pth")