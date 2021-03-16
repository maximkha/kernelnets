import torch.nn as nn
import torch
import math
import numpy as np

class MemLayer(nn.Module):
    def __init__(self, size_in:int, size_out:int, usebias:bool=True):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        self.usebias = usebias

        # with torch.no_grad():
        #     torch.fill_(self.weights, .5)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init



    def forward(self, x:torch.Tensor):
        # wx = torch.mm(x, self.weights.t())
        # return torch.add(wx, self.bias)  # w times x + b
        # wx = ((x - self.weights)**2).mean(dim=1)
        # print("X--")
        # print(x)
        
        wx = -((x.unsqueeze(0) - self.weights.unsqueeze(1))**2).mean(axis=-1).squeeze(1)
        
        # wx = ((x.unsqueeze(0) - self.weights.unsqueeze(1)).abs()).mean(axis=-1).squeeze(1) # WORSE
        # wx = (torch.relu(x.unsqueeze(0) - self.weights.unsqueeze(1))).mean(axis=-1).squeeze(1) # NOT SUPER GOOD
        # wx = (torch.hardshrink(x.unsqueeze(0) - self.weights.unsqueeze(1))).mean(axis=-1).squeeze(1)
        # print("WX--")
        # print(wx.squeeze)
        # print("SQ--")
        # print(x.unsqueeze(0) - self.weights.unsqueeze(1))
        # print("BIA--")
        # print(torch.add(wx, self.bias))
        
        #print(x.shape)
        #print(wx.transpose(0,1).shape)
        #print(self.bias.shape)

        # norm = torch.distributions.normal.Normal(0,1)
        # wx = (1-((norm.cdf(x.unsqueeze(0) - self.weights.unsqueeze(1))-.5).abs())*2).mean(axis=-1).squeeze(1) # ((x.unsqueeze(0) - self.weights.unsqueeze(1)).squeeze(1)
        
        if self.usebias: wx = torch.add(wx.transpose(0,1), self.bias)
        return wx

