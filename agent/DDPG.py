import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sample import ReplayBuffer
from network.body import policy_net, value_net
from utils.update import soft_update, hard_update


class DDPG:
    def __init__(self, input_channels: int, width: int, action_dim: int, noisy: bool, training: bool, gamma: float,
                 learning_rate: float, tau: float, batch_size: int, capacity: int, replace_interval: int,
                 checkpoint_dir=''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma  # discount factor
        self.batch_size = batch_size  # batch size
        self.replace_interval = replace_interval  # target net update interval
        self.checkpoint_dir = checkpoint_dir + 'DDPG/'

        self.policy_net = policy_net(input_channels, width, action_dim, noisy, training).to(self.device)
        self.target_policy_net = policy_net(input_channels, width, action_dim, noisy, training).to(self.device)
        self.value_net = value_net(input_channels, width, noisy, training).to(self.device)
        self.target_value_net = value_net(input_channels, width, noisy, training).to(self.device)

        self.policy_net_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity)
        self.learn_step_counter = 0
        # self.loss = nn.MSELoss()
        self.tau = tau

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self):
        if self.memory.__len__ < self.batch_size:
            return
        # hard_update每隔一段时间更新一次target_net，soft_update每次更新都更新一次target_net
        soft_update(self.target_policy_net, self.policy_net, tau=self.tau)
        soft_update(self.target_value_net, self.value_net, tau=self.tau)
        # if self.learn_step_counter % self.replace_interval == 0:
        #     hard_update(self.target_policy_net, self.policy_net)
        #     hard_update(self.target_value_net, self.value_net)
        self.learn_step_counter += 1

        stage, action, reward, next_stage, done = self.memory.sample(self.batch_size)
        stage = torch.from_numpy(stage).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_stage = torch.from_numpy(next_stage).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        target = reward + self.gamma * self.target_value_net(next_stage, self.target_policy_net(next_stage)) * (
                    1 - done)
        value_loss = F.smooth_l1_loss(self.value_net(stage, action), target.detach())  # 这里的损失函数也可以用MSE
        self.value_net_optimizer.zero_grad()
        value_loss.backward()
        self.value_net_optimizer.step()

        policy_loss = -self.value_net(stage, self.policy_net(stage)).mean()
        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()
