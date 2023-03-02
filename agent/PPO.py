import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from network.net import AC_Net
from utils.sample import ReplayBuffer
from utils.value import compute_target
from utils.update import hard_update


class PPO:
    def __init__(self, input_channels, width, action_dim, hidden_dim, batch_size, capacity, learning_rate, gamma,epoch,eps_clip=0.2,
                 checkpoint_dir=''):
        super(PPO, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
        self.action_dim = action_dim  # 动作维度
        self.model = AC_Net(input_channels=input_channels, width=width, action_dim=action_dim,
                            hidden_dim=hidden_dim)  # 新的模型
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # 折扣因子
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # 优化器
        self.memory = ReplayBuffer(capacity)  # 记忆库
        self.model_old = copy.deepcopy(self.model)  # 旧的模型
        self.batch_size = batch_size  # 批大小
        self.epoch = epoch  # 迭代次数
        self.eps_clip = eps_clip  # 裁剪系数
        self.model.to(self.device)
        self.model_old.to(self.device)
    def choose_action(self, stage):
        """
        input: stage(the shape of stage is (batch_size, input_channels, width, width))
        output: action(the shape of action is (batch_size, 1))
        """
        with torch.no_grad():
            action, log_prob, value = self.model_old.act(stage)
        return action
    
    def learn(self):
        if self.memory.size < self.batch_size:
            return
        stage, _, reward, _, done = tuple(map(lambda x:torch.from_numpy(np.array(x)).float().to(self.device),self.memory.sample(self.batch_size))) # 从memory中采样数据
        old_action, old_log_prob, old_value = self.model_old.act(stage) # 旧模型被固定住，并与环境交互进行采样
        eps = np.finfo(np.float32).eps.item()
        # 计算advantage
        R = compute_target(self.gamma, reward.reshape(-1).tolist(), done.reshape(-1).tolist()).to(self.device) # the shape of R is (batch_size,) 
        R = (R-R.mean())/(R.std()+eps) # 归一化

        advantage = R.detach() - old_value.detach().reshape(-1) # the shape of advantage is (batch_size,) old_value 是旧的模型计算出来的价值函数，即V(s)

        for _ in range(self.epoch):
            # 使用新的模型计算log_prob,value,entropy
            log_prob, value, entropy = self.model.evaluate(stage, old_action)
            # squeeze the shape of value from (batch_size,1) to (batch_size,)
            value = torch.squeeze(value)
            # 计算比率
            ratio = torch.exp(log_prob - old_log_prob.detach())
            surr1 = ratio * advantage # 实际计算过程用均值代替了期望
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage # PPO2
            # loss由三部分组成 1.策略梯度 2.价值函数 3.熵
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(value,R) - 0.01 * entropy
            print(loss)
            self.optimizer.zero_grad()
            loss.mean().backward() # 均值代替期望
            self.optimizer.step()
        
        hard_update(self.model_old, self.model)

if __name__ == "__main__":
    model = PPO(input_channels=4, width=224, action_dim=4, hidden_dim=512, batch_size=2, capacity=1000,epoch=1,
                learning_rate=0.001, gamma=0.99)
    # 向memory中添加数据
    for i in range(100):
        model.memory.push((np.random.randn(4, 224, 224), np.random.randn(1), np.random.randn(1), np.random.randn(4, 224, 224), np.random.randn(1)))
    model.learn()
