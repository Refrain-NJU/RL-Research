import torch
from torch import nn,Tensor,optim
import torch.nn.functional as F
import numpy as np
import sys,gym,time,random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.memory import ReplayBuffer

class Actor(nn.Module):
    def __init__(self,input_size,output_size):
        super(Actor,self).__init__()
        self.net=nn.Sequential(nn.Linear(input_size,400),
                               nn.ReLU()
                               nn.Linear(400,300)
                               nn.ReLU()
                               nn.Linear(300,output_size)
                               nn.Tanh())
       
    def forward(self,s):
        return self.net(s)

class Critic(nn.Module):
    def __init__(self,input_size,output_size):
        super(Critic,self).__init__()
        self.net=nn.Sequential(nn.Linear(input_size,400),
                               nn.ReLU()
                               nn.Linear(400,300)
                               nn.ReLU()
                               nn.Linear(300,output_size)
                               nn.Tanh())
     def forward(self,s,a):
         x=torch.cat([s,a],1)
         return self.net(x)

class DDPG(object):
    def __init(self):
        pass