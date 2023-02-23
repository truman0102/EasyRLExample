import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from network.body import Conv_block,DoubleHead_MLP_block
from utils.value import compute_target

class AC_Net(nn.Module):
    # batch_size, input_channels, width, width = input.shape
    def __init__(self, input_channels, width, action_dim, hidden_dim=512):
        super(AC_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc = DoubleHead_MLP_block(self.feature_dim, hidden_dim, action_dim)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x

    def v(self, x):
        x = self.feature(x)
        return self.fc.v(x)

    def a(self, x):
        x = self.feature(x)
        return self.fc.a(x)

class A2C:
    def __init__(self, input_channels, width, action_dim, hidden_dim, learning_rate, gamma, checkpoint_dir=''):
        super(A2C, self).__init__()
        self.action_dim = action_dim
        self.model = AC_Net(input_channels=input_channels, width=width, action_dim=action_dim, hidden_dim=hidden_dim)
        self.lr = learning_rate
        self.gamma = gamma
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, max_train_steps=10000, update_interval=5, print_interval=100):
        step_idx = 0
        # initialize environment, state, etc.
        s = np.random.random((1, 4, 224, 224))
        while step_idx < max_train_steps:
            s_list, a_list, r_list, done_list = [], [], [], []
            for _ in range(update_interval):
                # get data
                prob = torch.softmax(self.model.a(torch.from_numpy(s).float()), dim=1).reshape(-1)
                action = np.random.choice(range(self.action_dim), p=prob.detach().numpy())

                # s_prime, r, done, info = envs.step(action) # get s_next, r, done, info from action
                s_prime = np.random.random((1, 4, 224, 224))

                r = np.random.random()
                s_list.append(s)
                a_list.append(action)
                r_list.append(r)
                info = 0
                done_list.append(info)

                s = s_prime
                step_idx += 1
            # torch.from_numpy方法是浅复制，需要注意
            s_final = torch.from_numpy(s_prime).float()
            v_final = self.model.v(s_final).detach().clone().numpy().reshape(-1)

            # update_interval, 1 = td_target.shape
            td_target = compute_target(gamma=self.gamma, v_final=v_final, r_lst=r_list, done_lst=done_list)

            s_vec = torch.from_numpy(np.array(s_list)).float().squeeze(1)
            # update_interval,state_dim = s_vec.shape
            a_vec = torch.LongTensor(a_list).unsqueeze(1)
            # update_interval,action_dim = a_vec.shape
            advantage = td_target - self.model.v(s_vec)
            prob_actor = torch.softmax(self.model.a(s_vec), dim=-1)
            prob_actor_true = prob_actor.gather(1, a_vec)
            loss = -(torch.log(prob_actor_true) * advantage.detach()).mean() + \
                   F.smooth_l1_loss(self.model.v(s_vec).reshape(-1), td_target.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if step_idx % print_interval == 0:
                print(loss.item())
