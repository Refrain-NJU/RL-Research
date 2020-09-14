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
    def __init__(self,state_size,action_size):
        super(Actor,self).__init__()
        self.net=nn.Sequential(nn.Linear(state_size,400),
                               nn.ReLU()
                               nn.Linear(400,300)
                               nn.ReLU()
                               nn.Linear(300,action_size)
                               nn.Tanh())
       
    def forward(self,s):
        return self.net(s)

class Critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(Critic,self).__init__()
        self.net=nn.Sequential(nn.Linear(state_size+action_size,400),
                               nn.ReLU()
                               nn.Linear(400,300)
                               nn.ReLU()
                               nn.Linear(300,1)
                               nn.Tanh())
     def forward(self,s,a):
         x=torch.cat([s,a],1)
         return self.net(x)

class DDPG(object):
    def __init(self,state_size,action_size,gamma,epsilon,tau,replay_capacity,iters,batch_size,lr):
        self.state_size=state_size
        self.action_size=action_size
        self.gamma=gamma
        self.epsilon=epsilon
        self.tau=tau
        self.target_replace_iter=iters            
        self.learn_step_cnt=0
        self.batch_size=batch_size
        self.lr=lr
        self.reward_list=[]

        self.actor=Actor(self.state_size,self.action_size)
        self.actor_target=Actor(self.state_size,self.action_size)
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=self.lr)
        self.critic=Critic(self.state_size,self.action_size)
        self.critic_target=Critic(self.state_size,self.action_size)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=self.lr)
        self.mse_loss=nn.MSELoss()

    def choose_action(self,state):
        action=self.actor(state)
        print(action.shape)
        return action

    def learn(self):
        pass

    def train(self):
        pass