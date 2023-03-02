import torch
import torch.nn as nn
from torch.distributions import Categorical
from network.body import Conv_block, DoubleHead_MLP_block


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

    def act(self, stage):
        """
        the shape of prob is (batch_size,action_dim)
        input: stage
        output: action,log_prob,value
        """
        prob, value = self.forward(stage)
        prob = torch.softmax(prob, dim=1)
        m = Categorical(prob)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.detach(), log_prob.detach(), value.detach()

    def evaluate(self, stage, action):
        """
        input: stage,action
        output: log_prob,value,entropy
        """
        prob, value = self.forward(stage)
        prob = torch.softmax(prob, dim=1)
        m = Categorical(prob)
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return log_prob, value, entropy
