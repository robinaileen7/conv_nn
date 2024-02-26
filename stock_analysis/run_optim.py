import torch.optim as optim
import torch
from run_nn import model
import torch.nn as nn
import numpy as np
import pandas as pd

class run_optim:
    def __init__(self, alpha, size, X_y_set):
        self.optimizer = optim.SGD(model.parameters(), lr=alpha, momentum=0.75)
        self.colored = False
        self.size = size
        self.X_train = X_y_set[0]
        self.y_train = X_y_set[1]
        self.X_test = X_y_set[2]
        self.y_test = X_y_set[3]
        self.batch_size = round(len(self.X_train)/2)

    @staticmethod
    def loss_function(x):
        if x == 'MSE':
            return nn.MSELoss()
        elif x == 'cross entropy':
            return nn.CrossEntropyLoss()

    def model_train(self):
        optimizer = self.optimizer
        batch_size = self.batch_size
        X_train = self.X_train
        y_train = self.y_train

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
        X_test = self.X_test
        y_test = self.y_test

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

if __name__ == "__main__":
    from stock_performance import X_train, y_train, X_test, y_test
    X_y_set = X_train, y_train, X_test, y_test
    # Use size = 100 for images
    for i in [0.005, 0.007, 0.008, 0.009, 0.01]:
        run_optim(alpha=i, size=252, X_y_set=X_y_set).model_test()