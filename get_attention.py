from __future__ import print_function
import torch
from torch import nn
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# -------------------- Step 1: Define the function that receives the feature ---------------------- #
# Here, a class is defined, which has a function hook_fun that receives feature. 
# The class is defined to facilitate the extraction of multiple intermediate layers.
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        '''
        Note that the hook function used to handle features must contain three parameters [module, fea_in, fea_out].
        You can name the parameters as you like, but their meaning is fixed:
        The first parameter represents a sub-module in torch, such as Linear, Conv2d, etc.
        The second parameter is the input to the module, its type is a tuple;
        The third parameter is the output of the module, its type is tensor.
        Note that the input and output types are different, keep that in mind.
        '''
        self.fea = fea_out

# ---------- Step 2: Register the hook and tell the model at which layers I will extract features -------- #
def get_feas_by_hook(model):
    """
    To extract the feature after Conv2d, we need to iterate through the model's modules, find Conv2d,
    and register the hook function onto this module. 
    This is equivalent to telling the model that we want to handle the feature output of this layer 
    using hook_fun at the Conv2d layer.
    Since there may be multiple Conv2d layers in a model, we use hook_feas to store the feature 
    after each Conv2d layer.
    """
    fea_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Sigmoid):  # Find Sigmoid layers
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)  # Register the hook
            fea_hooks.append(cur_hook)

    return fea_hooks


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

# Set random seed
setup_seed(66)

# Data transformation pipeline
data_transform = transforms.Compose(
    [
        # transforms.Resize(32),  # Resize image, keeping the aspect ratio, shortest side is 32 pixels
        # transforms.CenterCrop(32),  # Center crop the image to 32*32 pixels
        transforms.ToTensor(),  # Convert Image to Tensor, normalizing to [0, 1]
        transforms.Normalize(mean=[0.408, 0.380, 0.340], std=[0.116, 0.137, 0.203])  # Normalize to [-1,1], specify mean and std
    ]
)

# Load the dataset
dataset = datasets.ImageFolder(root="datasets/2750", transform=data_transform)

# Split dataset into training and testing sets
train_size = int(0.7 * len(dataset))  # Training set is 70% of the dataset
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load checkpoint and model
checkpoint = torch.load("model_save/vgg16_SE_train_best.pth.tar")
model = torch.load("models/vgg16_SE.pth")
model.load_state_dict(checkpoint['state_dict'])

# model.to('cuda')  # Uncomment if using a GPU

n = 0
av = []
fea_hooks = get_feas_by_hook(model)  # Register hooks
long = len(fea_hooks)  # Number of hooks

# Iterate over dataset
for data in dataset:
        n = n + 1
        imgs, targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        out = model(imgs)  # Forward pass through the model
        print(fea_hooks[1].fea.shape)  # Print feature shape of the second hooked layer
        break
        if n == 1:
            # First batch
            for i in range(long):
                av.append(fea_hooks[i].fea)
        else:
            # Accumulate features for subsequent batches
            for i in range(long):
                av[i] = av[i] + fea_hooks[i].fea

# Calculate the average feature map
for i in range(len(av)):
    av[i] = torch.mean(av[i], axis=0) / (n-1)

# Save the average feature maps and count
torch.save(av, 'av_avg.pth')
torch.save(n, 'n.pth')
