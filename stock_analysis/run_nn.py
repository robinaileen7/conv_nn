import torch
dtype = torch.float
device = torch.device("cuda:0") 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.colored = False
        if self.colored:
            self.conv1 = nn.Conv2d(3, 24, 5)
        else:
            self.conv1 = nn.Conv2d(1, 24, 5)
        self.conv2 = nn.Conv2d(24, 48, 5)
        self.conv3 = nn.Conv2d(48, 96, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2,2))

        x = torch.flatten(x, 1)
        m1 = nn.Linear(x[0].shape[0], 100)
        m2 = nn.Linear(100, 2)
        x = F.relu(m1(x))
        x = m2(x)
        x = F.softmax(x, dim=1)
        return x

model = Net()

#print(model(X))
#print(y)