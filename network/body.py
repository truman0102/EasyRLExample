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


class MLP_block(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,noisy=False, training=False):
        super(MLP_block, self).__init__()
        if noisy:
            self.fc1 = FactorizedNoisyLinear(input_dim, hidden_dim, is_training=training)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, output_dim, is_training=training)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x

class DoubleHead_MLP_block(nn.Module):
    def __init__(self,input_dim,hidden_dim,action_dim,noisy=False,training=False):
        if noisy:
            self.fc1 = FactorizedNoisyLinear(input_dim, hidden_dim, is_training=training)
            self.fc2 = FactorizedNoisyLinear(hidden_dim, action_dim, is_training=training)
            self.fc3 = FactorizedNoisyLinear(hidden_dim, 1, is_training=training)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        advantage = self.fc2(x)
        value = self.fc3(x)
        return advantage,value
    
    def v(self,x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        return x
    
    def a(self,x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


class MLP_block_Q(nn.Module):
    """
    input: stage, action
    output: Q(s,a)
    """
    def __init__(self, input_dim,action_dim, hidden_dim_a,hidden_dim_s, output_dim=1, noisy=False, training=False):
        if noisy:
            self.fc1 = FactorizedNoisyLinear(input_dim, hidden_dim_s, is_training=training)
            self.fc2 = FactorizedNoisyLinear(action_dim,hidden_dim_a, is_training=training)
            self.fc3 = FactorizedNoisyLinear(hidden_dim_s+hidden_dim_a, output_dim, is_training=training)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim_s)
            self.fc2 = nn.Linear(action_dim, hidden_dim_a)
            self.fc3 = nn.Linear(hidden_dim_s+hidden_dim_a, output_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x, a):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        a = self.fc2(a)
        a = self.leaky_relu(a)
        x = torch.cat([x,a],dim=1)
        x = self.fc3(x)
        return x
        return torch.tanh(x)