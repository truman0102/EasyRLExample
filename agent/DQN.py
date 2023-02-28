import torch
import torch.nn as nn
import torch.nn.functional as F
from network.body import Conv_block, MLP_block, DoubleHead_MLP_block
from utils.sample import ReplayBuffer
from utils.update import hard_update

class DQN_Net(nn.Module):
    """
    input : stage
    output: Q(s,a) for all a
    batch_size, input_channels, width, width = input.shape
    """
    def __init__(self, input_channels, width, action_dim, hidden_dim, noisy=False, training=False):
        super(DQN_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc = MLP_block(self.feature_dim, hidden_dim, action_dim, noisy, training)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x


class Dueling_DQN_Net(nn.Module):
    # batch_size, input_channels, width, width = input.shape
    def __init__(self, input_channels, width, action_dim, hidden_dim, noisy=False, training=False):
        super(Dueling_DQN_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc = DoubleHead_MLP_block(self.feature_dim, hidden_dim, action_dim, noisy, training)

    def forward(self, x):
        x = self.feature(x)
        advantage, value = self.fc(x)
        return value + advantage - advantage.mean()

class DQN:
    def __init__(self, input_channels, width, action_dim, hidden_dim, batch_size, capacity, learning_rate, gamma,
                 e_greedy, replace_target_iter, noisy, eps_min=0.01, eps_dec=5e-7, checkpoint_dir=''):
        super(DQN, self).__init__()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.capacity = capacity
        self.gamma = gamma  # 折扣因子
        self.eplison = e_greedy  # 贪婪因子
        self.replace = replace_target_iter  # 更新目标网络的频率
        self.eps_min = eps_min  # 最小贪婪因子
        self.eps_dec = eps_dec  # 贪婪因子衰减
        self.checkpoint_dir = checkpoint_dir  # 模型保存路径
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(capacity)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.eval_net = DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
        self.target_net = DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def decrement_epsilon(self):
        self.eplison = self.eplison - self.eps_dec if self.eplison > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.__len__() < self.batch_size:  # 如果记忆库中的样本数小于batch_size，就不进行学习
            return
        self.learn_step_counter += 1  # 记录学习的次数
        if self.learn_step_counter % self.replace == 0:  # 每隔一段时间更新目标网络
            # 在第一次学习的时候，目标网络和评估网络的参数是一样的
            hard_update(self.target_net, self.eval_net)
        
        stage, action, reward, next_stage, done = tuple(map(lambda x:torch.from_numpy(x).float().to(self.device),self.memory.sample(self.batch_size)))

        # 根据评估网络预测当前状态下的Q值
        q_eval = self.eval_net.forward(stage).gather(1, action.unsqueeze(1)).squeeze(1)  # q_eval.shape = (batch_size,1)
        # 根据目标网络计算下一个状态下的Q值
        q_next = self.target_net.forward(next_stage).detach()  # q_next.shape = (batch_size,action_dim)
        # 根据当前状态和下一个状态的Q值计算目标Q值，取下一个状态下的最大Q值
        q_target = reward + self.gamma * torch.max(q_next, 1)[0]  # q_target.shape = (batch_size,1)
        # 计算损失
        loss = F.mse_loss(q_eval, q_target)  # 计算损失
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        self.decrement_epsilon()  # 贪婪因子衰减


class DQNs:
    # 包含DDQN和DuelingDQN
    def __init__(self, input_channels, width, action_dim, hidden_dim, batch_size, capacity, learning_rate, gamma,
                 e_greedy, replace_target_iter, noisy, DDQN=False, DuelingDQN=False, eps_min=0.01, eps_dec=5e-7,
                 checkpoint_dir=''):
        super(DQNs, self).__init__()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.capacity = capacity
        self.gamma = gamma  # 折扣因子
        self.eplison = e_greedy  # 贪婪因子
        self.replace = replace_target_iter  # 更新目标网络的频率
        self.eps_min = eps_min  # 最小贪婪因子
        self.eps_dec = eps_dec  # 贪婪因子衰减
        self.checkpoint_dir = checkpoint_dir  # 模型保存路径
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(capacity)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.DDQN = DDQN
        self.DuelingDQN = DuelingDQN
        if self.DuelingDQN:
            self.eval_net = Dueling_DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
            self.target_net = Dueling_DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
        else:
            self.eval_net = DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
            self.target_net = DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def decrement_epsilon(self):
        self.eplison = self.eplison - self.eps_dec if self.eplison > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.__len__() < self.batch_size:  # 如果记忆库中的样本数小于batch_size，就不进行学习
            return
        self.learn_step_counter += 1  # 记录学习的次数
        if self.learn_step_counter % self.replace == 0:  # 每隔一段时间更新目标网络
            # 在第一次学习的时候，目标网络和评估网络的参数是一样的
            hard_update(self.target_net, self.eval_net)
        stage, action, reward, next_stage, done = tuple(map(lambda x:torch.from_numpy(x).float().to(self.device),self.memory.sample(self.batch_size)))

        # 根据评估网络预测当前状态下的Q值
        q_eval = self.eval_net.forward(stage).gather(1, action.unsqueeze(1)).squeeze(1)
        if self.DDQN:
            argmax_a = self.eval_net.forward(next_stage).argmax(1).unsqueeze(1)  # 更新网络选取动作
            # 根据目标网络计算下一个状态下全部动作的Q值,并根据更新网络选取的动作，选取Q值
            q_next = self.target_net.forward(next_stage).gather(1, argmax_a).squeeze(1)
        else:
            # 根据目标网络计算下一个状态下的Q值，并选取最大的Q值
            q_next = self.target_net.forward(next_stage).detach().max(1)[0]  # 目标网络选取动作
        q_target = reward + self.gamma * q_next
        loss = F.mse_loss(q_eval, q_target)  # 计算损失
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        self.decrement_epsilon()  # 贪婪因子衰减
