import torch
from torch import nn,Tensor,optim
import torch.nn.functional as F
import numpy as np
import sys,gym,time,random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.memory import ReplayBuffer


env=gym.make('HalfCheetah-v2')
s=env.reset()
done=False
while not done:
    env.render()
    s_,r,done,_=env.step(env.action_space.sample())
    s=s_
env.close()