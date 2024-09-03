import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self, r1=1, r2=1):
        super(MyLoss, self).__init__()
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.r1 = r1
        self.r2 = r2

    def loss1(self, inputs, targets):
        x = torch.zeros_like(inputs)
        j = 0
        for i in targets:
            x[j][i] = 1
            j = j + 1

        inputs = self.softmax(inputs)
        d = torch.abs(inputs - x).mean(axis=1)
        loss = torch.exp(torch.pow(d, 1) + torch.pow(d, 2))
        return loss.mean()
        
    def forward(self, inputs, targets):

        return self.r1 * self.loss1(inputs, targets) + self.r2 * self.CrossEntropy(inputs, targets)   

