import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cv1 = nn.Conv2d(4, 32, 8, 4)
        self.cv2 = nn.Conv2d(32, 64, 4, 2)
        self.fc1 = nn.Linear(5184, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.cv1(x.reshape([-1, 4, 84, 84]))
        x = F.relu(x)
        x = self.cv2(x)
        x = F.relu(x)

        x = self.fc1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.fc2(x)
        return x
