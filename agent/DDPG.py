import torch
import torch.nn.functional as F
from utils.sample import ReplayBuffer
from network.actor import policy_net
from network.critic import value_net
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
        self.target_policy_net.eval()
        self.value_net = value_net(input_channels, width, noisy, training).to(self.device)
        self.target_value_net = value_net(input_channels, width, noisy, training).to(self.device)
        self.target_value_net.eval()

        self.policy_net_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity)
        self.learn_step_counter = 0
        # self.loss = nn.MSELoss()
        self.tau = tau # soft update parameter

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

        stage, action, reward, next_stage, done = tuple(map(lambda x:torch.from_numpy(x).float().to(self.device),self.memory.sample(self.batch_size)))

        # 用真实的奖励和下一步的Q来拟合当前的Q 然后让价值网络的输出逼近这个Q
        target = reward + self.gamma * self.target_value_net(next_stage, self.target_policy_net(next_stage)) * (1 - done)
        value_loss = F.smooth_l1_loss(self.value_net(stage, action), target.detach())  # 这里的损失函数也可以用MSE
        self.value_net_optimizer.zero_grad()
        value_loss.backward()
        self.value_net_optimizer.step()
        # 策略网络的目的是让价值网络的输出最大化，所以这里的损失函数是价值网络的输出的负值，输出越大损失越小
        policy_loss = -self.value_net(stage, self.policy_net(stage)).mean() # 取均值是因为要求的是一个batch的平均值
        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()
