from torch.distributions import Normal
import numpy as np
from network.layer import *
from network.body import Conv_block, MLP_block


class policy_net(nn.Module):
    """
    input: stage
    output: action distribution
    """

    def __init__(self, input_channels: int, width: int, action_dim: int, hidden_dim=512, noisy=False, training=False):
        super(policy_net, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc = MLP_block(self.feature_dim, hidden_dim,
                            action_dim, noisy, training)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x


class GaussianPolicy(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, input_channels: int, width: int, action_dim: int, max_action, hidden_dim=512):
        super(GaussianPolicy, self).__init__()
        self.feature = Conv_block(input_channels, width)
        self.feature_dim = self.feature.get_feature_dim()
        self.fc = MLP_block(self.feature_dim, hidden_dim, action_dim * 2)
        self.action_range = max_action

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        mu, log_std = torch.chunk(x, 2, dim=1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return mu, log_std

    def evaluate(self, x):
        '''
        generate sampled action with state as input wrt the policy network;
        deterministic evaluation provides better performance according to the original paper;
        '''
        mean, log_std = self.forward(x)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        z = Normal(0, 1).sample(mean.shape)
        # TanhNormal distribution as actions; reparameterization trick
        action_0 = torch.tanh(mean + std * z.to(self.device))
        action = self.action_range * action_0
        ''' stochastic evaluation '''
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - \
                   torch.log(1. - action_0.pow(2) + self.eps) - \
                   np.log(self.action_range)
        # loga+logb = log(ab)
        ''' deterministic evaluation '''
        # log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) -  np.log(self.action_range)
        '''
         both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
         the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
         needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
         '''
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std
