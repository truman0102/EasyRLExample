import numpy as np
from network.layer import *


class Conv_block(nn.Module):
    def __init__(self, input_channels, W, kernel_size=[8, 4, 3], stride=[4, 2, 1], padding=[0, 0, 0]):
        super(Conv_block, self).__init__()
        self.width = W
        self.conv1 = Conv2dLayer(input_channels, 32, 8, 4)
        self.conv2 = Conv2dLayer(32, 64, 4, 2)
        self.conv3 = Conv2dLayer(64, 64, 3, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        for i in range(3):
            # 向下取整
            self.width = math.floor((self.width - kernel_size[i] + 2 * padding[i]) / stride[i] + 1)  # output size

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        return x

    def get_feature_dim(self):
        return int(self.width * self.width * 64)


class policy_net(nn.Module):
    """
    input: stage
    output: action
    """

    def __init__(self, input_channels, width, action_dim, hidden_dim=512, noisy=False, trainging=False):
        super(policy_net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        if noisy:
            self.fc1 = FactorizedNoisyLinear(self.feature_dim, hidden_dim, trainging)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, action_dim, trainging)
        else:
            self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        action = self.fc2(x)
        return action


class value_net(nn.Module):
    """
    input:stage and action 
    output:Q(s,a)
    """

    def __init__(self, input_channels, width, action_dim, hidden_dim_a=64, hidden_dim_s=512, noisy=False,
                 trainging=False):
        super(value_net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        if noisy:
            self.fc1 = FactorizedNoisyLinear(self.feature_dim, hidden_dim_s, trainging)
            self.fc2 = FactorizedNoisyLinear(action_dim, hidden_dim_a, trainging)
            self.fc3 = FactorizedNoisyLinear(hidden_dim_s + hidden_dim_a, 1, trainging)
        else:
            self.fc1 = nn.Linear(self.feature_dim, hidden_dim_s)
            self.fc2 = nn.Linear(action_dim, hidden_dim_a)
            self.fc3 = nn.Linear(hidden_dim_s + hidden_dim_a, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, action):
        x = self.feature(x)
        x = self.fc1(x)
        a = self.fc2(action)
        x = torch.cat((x, a), dim=1)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        return x


class DQN_Net(nn.Module):
    # batch_size, input_channels, width, width = input.shape
    def __init__(self, input_channels, width, action_dim, hidden_dim, noisy=False, training=False):
        super(DQN_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        if noisy:
            self.fc1 = FactorizedNoisyLinear(self.feature_dim, hidden_dim, is_training=training)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, action_dim, is_training=training)
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
    # batch_size, input_channels, width, width = input.shape
    def __init__(self, input_channels, width, action_dim, hidden_dim, noisy=False, training=False):
        super(Dueling_DQN_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        if noisy:
            self.fc1 = FactorizedNoisyLinear(self.feature_dim, hidden_dim, is_training=training)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, action_dim, is_training=training)  # advantage
            self.fc3 = FactorizedNoisyLinear(hidden_dim, 1, is_training=training)  # value
        else:
            self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)  # advantage
            self.fc3 = nn.Linear(hidden_dim, 1)  # value
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        advantage = self.fc2(x)
        value = self.fc3(x)
        return value + advantage - advantage.mean()


class AC_Net(nn.Module):
    # batch_size, input_channels, width, width = input.shape
    def __init__(self, input_channels, width, action_dim, hidden_dim=512):
        super(AC_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc1 = nn.Linear(self.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # actor
        self.fc3 = nn.Linear(hidden_dim, 1)  # critic
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        action = self.fc2(x)
        value = self.fc3(x)
        return action, value

    def v(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        value = self.fc3(x)
        # batch_size,1 = value.shape
        return value

    def a(self, x):
        x = self.feature(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        action = self.fc2(x)
        # batch_size, action_dim = action.shape
        return action


class TD3_Critic_Net(nn.Module):
    def __init__(self, input_channels, width, action_dim, hidden_dim=256, noisy=False, trainging=False):
        super(TD3_Critic_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()

        # Q1
        if noisy:
            self.fc1 = FactorizedNoisyLinear(self.feature_dim + action_dim, hidden_dim, is_training=trainging)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, hidden_dim, is_training=trainging)
            self.fc3 = FactorizedNoisyLinear(hidden_dim, 1, is_training=trainging)
        else:
            self.fc1 = nn.Linear(self.feature_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
        # Q2
        if noisy:
            self.fc4 = FactorizedNoisyLinear(self.feature_dim + action_dim, hidden_dim, is_training=trainging)
            self.fc5 = FactorizedNoisyLinear(hidden_dim, hidden_dim, is_training=trainging)
            self.fc6 = FactorizedNoisyLinear(hidden_dim, 1, is_training=trainging)
        else:
            self.fc4 = nn.Linear(self.feature_dim + action_dim, hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, hidden_dim)
            self.fc6 = nn.Linear(hidden_dim, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, a):
        x = self.feature(x)
        x = torch.cat([x, a], dim=1)

        q1 = self.leaky_relu(self.fc1(x))
        q1 = self.leaky_relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = self.leaky_relu(self.fc4(x))
        q2 = self.leaky_relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def Q1(self, x, a):
        x = self.feature(x)
        x = torch.cat([x, a], dim=1)

        q1 = self.leaky_relu(self.fc1(x))
        q1 = self.leaky_relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1

    def Q2(self, x, a):
        x = self.feature(x)
        x = torch.cat([x, a], dim=1)

        q2 = self.leaky_relu(self.fc4(x))
        q2 = self.leaky_relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q2
