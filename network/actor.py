from network.layer import *
from network.body import Conv_block,MLP_block


class policy_net(nn.Module):
    """
    input: stage
    output: action distribution
    """
    def __init__(self, input_channels, width, action_dim, hidden_dim=512, noisy=False, trainging=False):
        super(policy_net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc = MLP_block(self.feature_dim, hidden_dim, action_dim, noisy, trainging)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x