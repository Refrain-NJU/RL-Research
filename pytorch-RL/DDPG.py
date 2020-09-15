import torch
from torch import nn,Tensor,optim
import torch.nn.functional as F
import numpy as np
import sys,gym,time,random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.memory import ReplayBuffer

class OUNoise:
    def __init__(self, mu, sigma=1.0, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
        
    def reset(self):
        self.x_prev = self.x0 if self.x0 else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu-self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class Actor(nn.Module):
    def __init__(self,state_size,action_size,max_action):
        super(Actor,self).__init__()
        self.max_action=max_action
        self.net=nn.Sequential(nn.Linear(state_size,400),
                               nn.ReLU(),
                               nn.Linear(400,300),
                               nn.ReLU(),
                               nn.Linear(300,action_size),
                               nn.Tanh())
       
    def forward(self,s):
        return self.max_action*self.net(s)

class Critic(nn.Module):
    def __init__(self,state_size,action_size):
        super(Critic,self).__init__()
        self.net=nn.Sequential(nn.Linear(state_size+action_size,400),
                               nn.ReLU(),
                               nn.Linear(400,300),
                               nn.ReLU(),
                               nn.Linear(300,1))
    def forward(self,s,a):
        x=torch.cat([s,a],1)
        return self.net(x)

class DDPG(object):
    def __init__(self,state_size,action_size,sigma=0.2,theta=0.15,gamma=0.99,tau=0.005,replay_capacity=1e6,batch_size=100,lr=1e-4):
        self.state_size=state_size
        self.action_size=action_size
        self.sigma=sigma
        self.theta=theta
        self.gamma=gamma
        self.tau=tau          
        self.batch_size=batch_size
        self.lr=lr
        self.reward_list=[]
        self.replay_buffer=ReplayBuffer(replay_capacity)
        self.actor=Actor(self.state_size,self.action_size,2)
        self.actor_target=Actor(self.state_size,self.action_size,2)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=self.lr*10)
        self.critic=Critic(self.state_size,self.action_size)
        self.critic_target=Critic(self.state_size,self.action_size)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=self.lr)
        self.mse_loss=nn.MSELoss()
        self.ou_noise=OUNoise(np.zeros(self.action_size),self.sigma,self.theta)

    def choose_action(self,state,add_noise):  #state should be a tensor
        if add_noise:
            action = self.actor(state) + Tensor(self.ou_noise())
        else:
            action=self.actor(state)   #should be like a 1D numpy-array [0.02]
        return action.detach().data.numpy()
    
    def store_transition(self,state,action,reward,next_state,done):
        self.replay_buffer.add(state,action,reward,next_state,done)
     
    def sample_transitions(self):
        batch=self.replay_buffer.sample(self.batch_size)
        b_s=torch.tensor([i.state for i in batch],dtype=torch.float32)
        b_a=torch.tensor([i.action for i in batch],dtype=torch.float32)
        b_r=torch.tensor([i.reward for i in batch],dtype=torch.float32)
        b_s_=torch.tensor([i.next_state for i in batch],dtype=torch.float32)
        b_d=torch.tensor([i.done for i in batch],dtype=torch.long)
        return b_s,b_a,b_r,b_s_,b_d
    
    def update_target_network(self,current_net,target_net,tau):
        '''target = (1-tau)*target+tau*current_net'''
        for param,target_param in zip(current_net.parameters(),target_net.parameters()):
            target_param.data.copy_((1-tau)*target_param.data + tau*param.data)

    def learn(self):
        b_s,b_a,b_r,b_s_,b_d=self.sample_transitions()
        target_q = self.critic_target(b_s_,self.actor_target(b_s_))
        target_q = b_r.view(-1,1)+(self.gamma*b_d.view(-1,1)*target_q).detach()
        current_q = self.critic(b_s,b_a)
        assert(target_q.shape==current_q.shape)
        
        critic_loss=self.mse_loss(current_q,target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -torch.mean(self.critic(b_s,self.actor(b_s)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.update_target_network(self.critic,self.critic_target,self.tau)
        self.update_target_network(self.actor,self.actor_target,self.tau)

    def evaluate(self,env,epoch):
        ep_r=0
        s=env.reset()
        while True:
            #env.render()
            a=self.choose_action(torch.FloatTensor(s),False)   
            s_,r,done,_=env.step(a)    #np.ndarray, np.float64
            ep_r+=r
            if done:
                self.reward_list.append(ep_r)
                print("epoch:%d,reward:%f"%(epoch,ep_r))
                break
            s=s_
        env.close()
    def train(self,epoch,env,begin_timestep):
        total_timestep=0
        for i in range(epoch):
            s=env.reset()
            ep_r=0
            while True:
                #env.render()
                if total_timestep<begin_timestep:
                    a=env.action_space.sample()
                else:
                    a=self.choose_action(torch.FloatTensor(s),True).clip(env.action_space.low,env.action_space.high)   
                s_,r,done,_=env.step(a)    #np.ndarray, np.float64
                ep_r+=r
                done_bool = 0 if done else 1
                self.store_transition(s,a,r,s_,done_bool)
                if total_timestep>begin_timestep:
                    self.learn()
                if done:
                    self.reward_list.append(ep_r)
                    break
                s=s_
                total_timestep+=1
                if i>170:
                    env.render()
            print("total_timestep:%d,epoch:%d,reward:%f"%(total_timestep,i,ep_r))
        env.close()



if __name__=="__main__":
    env=gym.make("Pendulum-v0")
    ACTION_SIZE=env.action_space.shape[0]
    STATE_SIZE=env.observation_space.shape[0]
    ddpg=DDPG(STATE_SIZE,ACTION_SIZE)
    ddpg.train(2000,env,1e4)

