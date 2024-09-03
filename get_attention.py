from __future__ import print_function
import torch
from torch import nn
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# -------------------- 第一步：定义接收feature的函数 ---------------------- #
# 这里定义了一个类，类有一个接收feature的函数hook_fun。定义类是为了方便提取多个中间层。
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_out

# ---------- 第二步：注册hook，告诉模型我将在哪些层提取feature -------- #
def get_feas_by_hook(model):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    fea_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Sigmoid):
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
# 设置随机数种子
setup_seed(66)

data_transform = transforms.Compose(
    [
        # transforms.Resize(32),  # 缩放图片，长宽比不变，最短边为32像素
        # transforms.CenterCrop(32),  # 从图片中间切出32*32的图片
        transforms.ToTensor(),  # 将图片（Image）转换成Tensor,归一化至[0，1]
        transforms.Normalize(mean=[0.408, 0.380, 0.340], std=[0.116, 0.137, 0.203])  # 正则化操作，标准化至[-1,1],规定均值和标准差
    ]
)

dataset = datasets.ImageFolder(root="datasets/2750", transform=data_transform)

# 划分训练集和测试集
train_size = int(0.7 * len(dataset))  # 训练集大小为数据集的70%
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


checkpoint = torch.load("model_save/vgg16_SE_train_best.pth.tar")
model = torch.load("models/vgg16_SE.pth")
model.load_state_dict(checkpoint['state_dict'])
# model.to('cuda')
n = 0
av = []
fea_hooks = get_feas_by_hook(model)
long = len(fea_hooks)
for data in dataset:
        n = n+1
        imgs, targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        out = model(imgs)
        print(fea_hooks[1].fea.shape)
        break
        if n==1:
            # out = model(imgs)
            for i in range(long):
                    av.append(fea_hooks[i].fea)
        else:
            # out = model(imgs)
            for i in range(long):
                    av[i] = av[i]+fea_hooks[i].fea

for i in range(len(av)):
    av[i] = torch.mean(av[i], axis=0)/(n-1)
torch.save(av, 'av_avg.pth')
torch.save(n, 'n.pth')


