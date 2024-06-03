import torch
dtype = torch.float
device = torch.device("cuda:0") # Uncommon this to run on GPU
# device = torch.device("cpu") # Uncommon this to run on CPU
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.colored = False
        self.out_channel_1 = 24
        self.out_channel_2 = 48
        self.out_channel_3 = 96
        self.kernel_size_1 = (7,1)
        self.kernel_size_2 = (1,7)
        self.kernel_size_3 = 7
        if self.colored:
            self.conv1 = nn.Conv2d(3, self.out_channel_1, self.kernel_size_1)
        else:
            self.conv1 = nn.Conv2d(1, self.out_channel_1, self.kernel_size_1)
        self.conv2 = nn.Conv2d(self.out_channel_1, self.out_channel_2, self.kernel_size_2)
        self.conv3 = nn.Conv2d(self.out_channel_2, self.out_channel_3, self.kernel_size_3)
        #self.dropout = nn.Dropout(0.0001)

    @staticmethod
    def activation_func(type, x):
        if type == 'tanh':
            return F.max_pool2d(F.tanh(x), (2,1))
        elif type == 'relu':
            return F.max_pool2d(F.relu(x), (2,1))
        elif type == 'sigmoid':
            #print(F.sigmoid(x).size())
            #print(F.max_pool2d(F.sigmoid(x), (2,2)).size())
            return F.sigmoid(x)

    def forward(self, x):
        x = F.max_pool2d(self.activation_func('sigmoid',self.conv1(x)), (2,1))
        x = F.max_pool2d(self.activation_func('sigmoid',self.conv2(x)), (2,1))
        x = self.activation_func('sigmoid',self.conv3(x))
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