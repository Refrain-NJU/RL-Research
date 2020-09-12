import torch
from torch import nn,Tensor,optim
import torch.nn.functional as F
import numpy as np
import sys,gym,time,random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.memory import ReplayBuffer

class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_size,256)
        self.fc2=nn.Linear(256,84)
        self.out=nn.Linear(84,output_size)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        out=self.out(x)
        return out

env=gym.make("CartPole-v0").unwrapped
ACTION_SIZE=env.action_space.n
STATE_SIZE=env.observation_space.shape[0]

class DoubleDQN():
    def __init__(self,gamma,epsilon,replay_capacity,iters,batch_size,lr):
        self.gamma=gamma
        self.epsilon=epsilon
        self.target_replace_iter=iters            
        self.learn_step_cnt=0
        self.batch_size=batch_size
        self.lr=lr
        self.reward_list=[]
        self.policy_net,self.target_net=Net(STATE_SIZE,ACTION_SIZE),Net(STATE_SIZE,ACTION_SIZE)
        self.replay_buffer=ReplayBuffer(replay_capacity)
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=self.lr)
        self.loss_func=nn.MSELoss()

    def choose_action(self,state):
        if random.random()<self.epsilon:
            action=np.random.randint(0,ACTION_SIZE)
            return action
        else:
            state=torch.unsqueeze(torch.Tensor(state),0)     #[1,4]
            action_value=self.policy_net.forward(state)
            action=torch.max(action_value,dim=1)[1].data.numpy()[0]  #torch.max(x)返回最大的一个值，torch.max(x,dim)返回一个namedturple，
            return action     #(value,indices)，dim=0表示从每一列中选最大的，dim=1表示从每一行中选最大的
    
    def store_transition(self,state,action,reward,next_state):
        self.replay_buffer.add(state,action,reward,next_state)
     
    def sample_transitions(self):
        batch=self.replay_buffer.sample(self.batch_size)
        b_s=torch.tensor([i.state for i in batch],dtype=torch.float32)
        b_a=torch.tensor([i.action for i in batch],dtype=torch.long)
        b_r=torch.tensor([i.reward for i in batch],dtype=torch.float32)
        b_s_=torch.tensor([i.next_state for i in batch],dtype=torch.float32)
        return b_s,b_a,b_r,b_s_
    
    def learn(self):
        if self.learn_step_cnt%self.target_replace_iter==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_cnt+=1
        b_s,b_a,b_r,b_s_=self.sample_transitions()           #将tensor([1,2,3])转换为tensor([[1],[2],[3]])
        q_val=self.policy_net(b_s).gather(1,b_a.view(-1,1))  #按列gather属于b_a的元素
        '''与dqn的区别在于，在计算标签y时，先使用policynet计算出下一个状态的max动作值
           然后用targetnet来评估这些动作值，结合reward形成标签y
        '''
        action_max=self.policy_net(b_s_).max(dim=1)[1]
        q_next=self.target_net(b_s_).detach().gather(1,action_max.view(-1,1))
        q_target=b_r.view(-1,1)+self.gamma*q_next
        assert(q_val.shape==q_target.shape)
        loss=self.loss_func(q_val,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self,epoch):
        print("collecting data...")
        for i in range(epoch):
            s=env.reset()
            ep_r=0
            while True:
                #env.render()
                a=self.choose_action(torch.FloatTensor(s))
                s_,r,done,_=env.step(a)
                
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2
                ep_r+=r
                
                self.store_transition(s,a,r,s_)
                if self.replay_buffer.is_full():
                    self.learn()
                    if done:
                        print("iter:%d,reward:%f"%(i,ep_r))
                if done:
                    self.reward_list.append(ep_r)
                    break
                s=s_
            
        env.close()

    def draw(self):
        plt.plot(self.reward_list)
        plt.title('DQN:CartPole-v0')
        plt.xlabel('epoch')
        plt.ylabel('reward')
        plt.savefig('./a.jpg')

if __name__=="__main__":
    ddqn=DoubleDQN(0.9,0.1,2000,100,32,0.01)
    ddqn.train(300)
    ddqn.draw()
