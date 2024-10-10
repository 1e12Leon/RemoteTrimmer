from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import shutil
import math  # init
import torchvision
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import numpy as np
import random
from Adaptive_Mining_Loss import MyLoss
from resnet18_SE import resnet18_SE

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
# random number seed
setup_seed(66)


data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.408, 0.380, 0.340], std=[0.116, 0.137, 0.203])
    ]
)

dataset = datasets.ImageFolder(root="datasets/EuroSAT", transform=data_transform)

# training set and test set
train_size = int(0.7 * len(dataset)) 
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Data Loader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# network model
rate = "0.7"
name = "resnet18_SE"
path = "models/resnet18_SE/"+name+"("+rate+").pth"
model = torch.load(path)

if torch.cuda.is_available():
    model = model.cuda()

# loss function
loss_fn = nn.CrossEntropyLoss()
# loss_fn = MyLoss(r1=1, r2=1)
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# Optimizer
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Set training parameters
# Record training times
total_train_step = 0
# Record testing times
total_test_step = 0
# Number of training epochs
epoch = 40

def train():
    model.train()
    total_train_step = 0
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        
        output = model(imgs)
        loss = loss_fn(output, targets)
        
        # Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record training times
        total_train_step = total_train_step + 1
        
        if total_train_step % 100 == 0:
            print("training times：{}, Loss：{}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

def test():
    # Test step starts
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            
            # accuracy
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            
    print("Loss on the whole test set：{}".format(total_test_loss))
    print("Accuracy on the whole test set{}".format(total_accuracy/float(len(test_dataloader.dataset))))
    return total_accuracy / float(len(test_dataloader.dataset))

path2 = "model_save/"+name+"/train_best("+rate+")(epoch=40).pth.tar"
def save_checkpoint(state, is_best, filename='temp_model/train.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, path2)

if __name__ == '__main__':
    # freeze_support()
    best_prec1 = 0.
    for i in range(0, epoch):
        print("-----The {}th round of training begins-----".format(i+1))
        if i in [epoch*0.5, epoch*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train()
        prec1 = test()
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
