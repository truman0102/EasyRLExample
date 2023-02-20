from layer import *


class Conv_block(nn.Module):
    def __init__(self, in_channels):
        super(Conv_block, self).__init__()
        self.conv1 = Conv2dLayer(in_channels, 32, 8, 4)
        self.conv2 = Conv2dLayer(32, 64, 4, 2)
        self.conv3 = Conv2dLayer(64, 64, 3, 1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        return x


class DQN_Net():
    # input:image
    def __init__(self):
        super(DQN_Net, self).__init__()
    pass


class DDQN_Net():
    # input:image
    pass


class Dueling_DQN_Net():
    # input:image
    pass


if __name__ == '__main__':
    # test
    x = torch.randn(1, 4, 224, 224)
    conv = Conv_block(4)
    y = conv(x)
    print(y.shape)
