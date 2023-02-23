import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sample import ReplayBuffer
from network.actor import GaussianPolicy
from network.critic import Twin_Value_Net

class SAC:
    def __init__(self,input_channels:int,width:int,action_dim:int,noisy:bool,gamma:float,learning_rate: float, tau: float, alpha:float,batch_size: int, capacity: int, replace_interval: int,checkpoint_dir=''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma # reward discount
        self.memory = ReplayBuffer(capacity)
        self.tau = tau # soft update parameter
        self.alpha = alpha # entropy temperature
        self.batch_size = batch_size
        self.replace_interval = replace_interval # 每隔多少步更新一次target网络
        self.learn_step_counter = 0

        self.actor = GaussianPolicy(input_channels=input_channels,width=width,action_dim=action_dim).to(self.device)
        self.critic = Twin_Value_Net(input_channels=input_channels, width=width, action_dim=action_dim, noisy=noisy).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) # 

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=learning_rate)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def train(self):
        if self.memory.__len__ < self.batch_size:
            return
        stage, action, reward, next_stage, done = self.memory.sample(self.batch_size)
        stage = torch.from_numpy(stage).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_stage = torch.from_numpy(next_stage).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        
    

