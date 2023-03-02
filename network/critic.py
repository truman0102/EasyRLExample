from network.layer import *
from network.body import Conv_block, MLP_block


class value_net(nn.Module):
    """
    input:stage and action 
    output:Q(s,a)
    """

    def __init__(self, input_channels, width, action_dim, hidden_dim_a=64, hidden_dim_s=512, noisy=False,
                 training=False):
        super(value_net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc = MLP_block(self.feature_dim + action_dim, hidden_dim_s + hidden_dim_a, 1, noisy, training)

    def forward(self, x, action):
        x = self.feature(x)
        x = torch.cat([x, action], dim=1)
        x = self.fc(x)
        return x


class Twin_Value_Net(nn.Module):
    """
    input:stage and action
    output:Q1(s,a) and Q2(s,a)
    """
    def __init__(self, input_channels, width, action_dim, hidden_dim=256, noisy=False, training=False):
        super(Twin_Value_Net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        # Q1
        self.fc1 = MLP_block(self.feature_dim + action_dim, hidden_dim, 1, noisy, training)
        # Q2
        self.fc2 = MLP_block(self.feature_dim + action_dim, hidden_dim, 1, noisy, training)

    def forward(self, x, a):
        x = self.feature(x)
        x = torch.cat([x, a], dim=1)
        return self.fc1(x), self.fc2(x)

    def Q1(self, x, a):
        x = self.feature(x)
        x = torch.cat([x, a], dim=1)
        return self.fc1(x)

    def Q2(self, x, a):
        x = self.feature(x)
        x = torch.cat([x, a], dim=1)
        return self.fc2(x)
