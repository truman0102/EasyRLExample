import copy
import torch
import torch.nn.functional as F
from utils.sample import ReplayBuffer
from network.actor import GaussianPolicy
from network.critic import Twin_Value_Net
from utils.update import soft_update

class SAC:
    def __init__(self,input_channels:int,width:int,action_dim:int,noisy:bool,gamma:float,learning_rate: float, tau: float, alpha:float,batch_size: int, capacity: int, replace_interval: int,target_entropy=None,checkpoint_dir=''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma # reward discount
        self.memory = ReplayBuffer(capacity)
        self.tau = tau # soft update parameter
        self.alpha = alpha # entropy temperature
        self.batch_size = batch_size
        self.replace_interval = replace_interval # 每隔多少步更新一次target网络
        self.learn_step_counter = 0
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.actor = GaussianPolicy(input_channels=input_channels,width=width,action_dim=action_dim).to(self.device)
        self.critic = Twin_Value_Net(input_channels=input_channels, width=width, action_dim=action_dim, noisy=noisy).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()
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
        self.learn_step_counter+=1
        stage, action, reward, next_stage, done = tuple(map(lambda x:torch.from_numpy(x).float().to(self.device),self.memory.sample(self.batch_size)))

        with torch.no_grad():
            new_action, log_prob, _,_,_ = self.actor.evaluate(next_stage)
            q1_target, q2_target = self.critic_target(next_stage,new_action)
            q_target = torch.min(q1_target,q2_target)
            value_target = reward + (1.0 - done) * self.gamma * (q_target - self.alpha * log_prob)
        q1,q2 = self.critic(stage,action)
        q_loss = F.smooth_l1_loss(q1,value_target)+F.smooth_l1_loss(q2,value_target)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        for para in self.critic.parameters():
            para.requires_grad = False

        eva_action, log_prob, _, _, _ = self.actor(stage)
        q1_eva,q2_eva = self.critic(stage,eva_action) # 当前时刻的状态和当前策略估计的动作
        q_pi = torch.min(q1_eva,q2_eva)
        policy_loss = (self.alpha*log_prob-q_pi).mean()
        alpha_loss = -self.alpha*(log_prob+self.target_entropy).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        for para in self.critic.parameters():
            para.requires_grad = True
        
        if self.learn_step_counter%self.replace_interval==0:
            soft_update(target=self.critic_target,source=self.critic,tau=self.tau)




    

