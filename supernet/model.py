import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockSum(nn.Module):
    def __init__(self, conv_channels=16):
        super(BlockSum, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(1, conv_channels, 5, 1, 2)
        self.choice = torch.zeros((1,))

    def set_choice(self, x):
        self.choice = x

    def forward(self, x):
        x = self.conv1(x) * (1 - self.choice) + self.conv2(x) * self.choice
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        return x


class BlockConcat(nn.Module):
    def __init__(self, conv_channels=16):
        super(BlockConcat, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(1, conv_channels, 5, 1, 2)
        self.conv_1x1 = nn.Conv2d(2 * conv_channels, conv_channels, 1, 1, 0)
        self.choice = torch.zeros((1,))

    def set_choice(self, x):
        self.choice = x

    def forward(self, x):
        x = torch.cat((self.conv11(x)*(1-self.choice), self.conv12(x)*self.choice), dim=1)
        x = self.conv1_1x1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        return x


class SuperNet(nn.Module):
    def __init__(self, conv1channels=16, conv2channels=16, hidden=64):
        super(SuperNet, self).__init__()
        self.conv11 = nn.Conv2d(1, conv1channels, 3, 1, 1)
        self.conv12 = nn.Conv2d(1, conv1channels, 5, 1, 2)
        self.conv21 = nn.Conv2d(conv1channels, conv2channels, 3, 1, 1)
        self.conv22 = nn.Conv2d(conv1channels, conv2channels, 5, 1, 2)
        self.fc1 = nn.Linear(2 * conv2channels * 49, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.choice = torch.zeros((2,))

    def set_subnet(self, x):
        self.choice = x

    def sample_subnet(self):
        self.set_subnet(torch.randint(2, (2,)))

    def forward(self, x):
        if self.choice[0] == 0:
            x = self.conv11(x)
        else:
            x = self.conv12(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.cat((self.conv21(x) * (1 - self.choice[1]), self.conv22(x) * self.choice[1]), dim=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class SuperNetSum(nn.Module):
    def __init__(self, conv1channels=16, conv2channels=16, hidden=64):
        super(SuperNetSum, self).__init__()
        self.block1 = BlockSum(conv1channels)
        self.block2 = BlockSum(conv2channels)
        self.fc1 = nn.Linear(conv2channels * 49, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.choice = torch.zeros((2,))

    def set_subnet(self, x):
        self.choice = x

    def sample_subnet(self):
        self.set_choice(torch.randint(2, (2,)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class SuperNetConcat(SuperNetSum):
    def __init__(self, conv1channels=16, conv2channels=16, hidden=64):
        super(SuperNetConcat, self).__init__(conv1channels, conv2channels, hidden)
        self.block1 = BlockConcat(conv1channels)
        self.block2 = BlockConcat(conv2channels)

