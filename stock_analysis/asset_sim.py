import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from run_optim import run_optim

class BS_sim:
    def __init__(self, r, sigma, g, S_0, T, t, time_step, N):
        self.r = r
        self.sigma = sigma
        self.g = g
        self.S_0 = S_0
        self.T = T
        self.t = t
        self.time_step = time_step
        self.N = N

    def dt(self):
        return 1/self.time_step
    
    def exact(self):
        r = self.r 
        sigma = self.sigma
        g = self.g
        S_0 = self.S_0
        T = self.T
        t = self.t
        time_step = self.time_step
        N = self.N

        n_path = round((T - t) * time_step) + 1
        S_array = np.zeros((N, n_path))
        S_array[:, 0] = np.log(S_0)
        np.random.seed(10)
        for i in range(1, n_path):
            S_array[:, i] = S_array[:, i - 1]  + ((r - g) - (1/2) * sigma ** 2) * self.dt() + np.random.normal(0, 1, N) * sigma * self.dt() ** (1/2)
        return np.exp(S_array)

time_step = 252
N = 1000
obj = BS_sim(r = 0.02, sigma = 0.8, g = 0.01, S_0 = 400, T = 1, t = 0, time_step = time_step, N = N)
# print(obj.exact().mean(axis = 0))
a = obj.exact()
_a = [x[-1]/x[0]-1 for x in a]
y = np.array([[0, 1] if x >= 0 else [1, 0] for x in _a])
y = torch.Tensor(y)

X_sim = torch.Tensor([])

for m in range(0, N):
    b = np.empty((time_step + 1, time_step + 1,))
    b[:] = np.nan
    b[0] = a[m]

    for i in range(1, time_step + 1):
        b[i][i:] = [y/x-1 for x, y in zip(a[m][:-1], a[m][i:])]

    cutoff = math.ceil((time_step + 1)/2)
    c = np.empty((cutoff-1, cutoff-1,))
    c[:] = np.nan
    for i in range(0, cutoff-1):
        c[i] = b[i+1][-cutoff:-1]

    # x is the square part of the upper triangular matrix 
    x = torch.Tensor(np.array(c)).view(-1, 1, cutoff-1, cutoff-1)
    X_sim = torch.cat((X_sim,x), 0)

train_split = round(len(X_sim) * 0.75)
X_train, X_test = X_sim[:train_split], X_sim[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

if __name__ == "__main__":
    #print(b[0], b[0][-cutoff:], b[0][-cutoff:-1])
    #print(X_train[0], y_train[0])
    X_y_set = X_train, y_train, X_test, y_test
    for i in [0, 0.5, 0.9]:
        run_optim(alpha=0.001, miu=i, size=cutoff-1, X_y_set=X_y_set).model_test()
 