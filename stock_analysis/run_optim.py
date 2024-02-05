import torch.optim as optim
import torch
from run_nn import model
import torch.nn as nn
from read_img import X_train, y_train, X_test, y_test
import numpy as np

class run_optim:
    def __init__(self, alpha):
        self.optimizer = optim.Adam(model.parameters(), lr=alpha)
        self.batch_size = round(len(X_train)/2)
        self.colored = False
        self.size = 100

    @staticmethod
    def loss_function(x):
        if x == 'MSE':
            return nn.MSELoss()
        elif x == 'cross entropy':
            return nn.CrossEntropyLoss()

    def model_train(self):
        optimizer = self.optimizer
        batch_size = self.batch_size

        for i in range(0, len(X_train)-batch_size):
            optimizer.zero_grad()
            loss = run_optim.loss_function('MSE')(model(X_train[i:i+batch_size]), y_train[i:i+batch_size])
            loss.backward()
            optimizer.step()
        print("Loss for training set is")
        print(loss)

    def model_test(self):
        run_optim.model_train(self)
        size = self.size
        correct = 0
        with torch.no_grad():
            for i in range(0, len(X_test)):
                if self.colored:
                    y_hat = model(torch.Tensor(X_test[i]).view(-1, 3, size, size))
                else:
                    y_hat = model(torch.Tensor(X_test[i]).view(-1, 1, size, size))
                y = torch.Tensor(y_test[i])
                if torch.argmax(y_hat) == torch.argmax(y):
                    correct += 1
        print("The accuracy is")
        print(correct/len(y_test))

for i in np.arange(0.01, 0.017, 0.001):
    print('tuning parameter: ' + str(i))
    run_optim(alpha=i).model_test()