import torch
from network.body import DQN_Net
from utils.sample import ReplayBuffer


class DDQN:
    def __init__(self, input_channels, width, action_dim, hidden_dim, batch_size, capacity, learning_rate, gamma,
                 e_greedy, replace_target_iter, noisy, eps_min=0.01, eps_dec=5e-7, checkpoint_dir=''):
        super(DDQN, self).__init__()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.capacity = capacity
        self.gamma = gamma
        self.eplison = e_greedy
        self.replace = replace_target_iter
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(capacity)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.eval_net = DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
        self.target_net = DQN_Net(input_channels, width, action_dim, hidden_dim, noisy).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

    def decrement_epsilon(self):
        self.eplison = self.eplison - self.eps_dec if self.eplison > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.__len__() < self.batch_size:  # 如果记忆库中的样本数小于batch_size，就不进行学习
            return
        self.learn_step_counter += 1  # 记录学习的次数
        if self.learn_step_counter % self.replace == 0:  # 每隔一段时间更新目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        states, actions, rewards, next_states = self.memory.sample_buffer(self.batch_size)  # 从记忆库中随机抽取batch_size个样本
        # tensor默认是从numpy转换过来的，memory中的数据需要是numpy格式
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        q_eval = self.eval_net.forward(states).gather(1, actions.unsqueeze(1)).squeeze(
            1)  # q_eval.shape = (batch_size,1)
        argmax_a = self.eval_net.forward(next_states).argmax(1).unsqueeze(1)  # argmax_a.shape = (batch_size,1)
        q_next = self.target_net.forward(next_states).gather(1, argmax_a).squeeze(1)  # q_next.shape = (batch_size,1)
        q_target = rewards + self.gamma * q_next
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decrement_epsilon()
