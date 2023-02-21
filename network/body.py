import numpy as np
from layer import *


class Conv_block(nn.Module):
    def __init__(self, input_channels, W):
        super(Conv_block, self).__init__()
        self.width = W
        self.conv1 = Conv2dLayer(input_channels, 32, 8, 4)
        self.conv2 = Conv2dLayer(32, 64, 4, 2)
        self.conv3 = Conv2dLayer(64, 64, 3, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        ks = [8, 4, 3]  # kernel size
        s = [4, 2, 1]  # stride
        p = [0, 0, 0]  # padding
        for i in range(3):
            # 向下取整
            self.width = math.floor(
                (self.width - ks[i] + 2 * p[i]) / s[i] + 1)  # output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        return x

    def get_feature_dim(self):
        return int(self.width * self.width * 64)


class DQN_Net(nn.Module):
    # input:image
    def __init__(self, input_channels, width, action_dim, hidden_dim, noisy=False):
        super(DQN_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        if noisy:
            self.fc1 = FactorizedNoisyLinear(self.feature_dim, hidden_dim)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, action_dim)
        else:
            self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


class Dueling_DQN_Net(nn.Module):
    # input:image
    def __init__(self, input_channels, width, action_dim, hidden_dim, noisy=False):
        super(Dueling_DQN_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        if noisy:
            self.fc1 = FactorizedNoisyLinear(self.feature_dim, hidden_dim)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, action_dim) # advantage
            self.fc3 = FactorizedNoisyLinear(hidden_dim, 1) # value
        else:
            self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim) # advantage
            self.fc3 = nn.Linear(hidden_dim, 1) # value
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        advantage = self.fc2(x)
        value = self.fc3(x)
        return value + advantage - advantage.mean()

class A2C_Net(nn.module):
    def __init__(self, input_channels, width, action_dim, hidden_dim):
        super(A2C_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim) # actor
        self.fc3 = nn.Linear(hidden_dim, 1) # critic
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        action = self.fc2(x)
        value = self.fc3(x)
        return action, value

if __name__ == '__main__':
    # test
    x = torch.randn(2, 4, 224, 224)
    net = Dueling_DQN_Net(4, 224, 6, 512)
    y = net(x)
    print(y.shape)
