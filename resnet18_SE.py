import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

##resnet每个残差链接模块
class BasicBlock(nn.Module):
    def __init__(self,inplanes: int,planes: int,stride: int = 1,downsample = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se1 = SELayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se2 = SELayer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.se1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,layers: list, num_classes: int,zero_init_residual: bool = False) -> None:
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se1 = SELayer(channel=self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.do = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, planes: int, blocks: int,stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        #先做7x7的卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.se1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.do(x)
        x = self.fc(x)

        return x


def resnet18_SE(num_classes, pretrained: bool = False, path= None, progress: bool = True) -> ResNet:
    '''
    pretrained决定是否用预训练的参数
    如果没有指定自己预训练模型的路径，就从官方网站下载resnet18-f37072fd.pth
    progress显示下载进度
    '''
    model=ResNet([2, 2, 2, 2], num_classes, zero_init_residual= True)
    if pretrained:
        if not path:
            pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth',progress=progress)
        else:
            pretrained_state_dict = torch.load(path)
        state_dict=model.state_dict()
        for k in pretrained_state_dict:
            if k in state_dict and k not in ['fc.weight', 'fc.bias']:
                state_dict[k] = pretrained_state_dict[k].data
        model.load_state_dict(state_dict)
    return model
