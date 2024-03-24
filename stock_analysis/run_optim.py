import torch.optim as optim
import torch
from run_nn import model
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")
from data_prep import data_yf
import config

class run_optim:
    def __init__(self, alpha, miu, size, X_y_set):
        self.optimizer = optim.SGD(model.parameters(), lr=alpha, momentum=miu)
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
                    y_hat = model(torch.Tensor(X_test[i]).view(-1, 3, size-1, size-1))
                else:
                    y_hat = model(torch.Tensor(X_test[i]).view(-1, 1, size-1, size-1))
                y = torch.Tensor(y_test[i])
                if torch.argmax(y_hat) == torch.argmax(y):
                    correct += 1
        print("The accuracy is")
        print(correct/len(y_test))
        return correct/len(y_test)

class myStruct():
    pass

myData = myStruct()
myData.ticker = config.ticker
myData.period = config.shift_period
myData.df_len = config.df_len

if __name__ == "__main__":
    date_list = pd.date_range(datetime.datetime(2010, 1, 1), datetime.datetime(2022, 1, 1), freq='Y')
    ls_ls = []
    for j in date_list:
        print(j)
        start = j
        end = start + relativedelta(years=2)
        temp = yf.download('MCD', str(start)[:10], str(end)[:10], progress = False)
        if len(temp) > myData.df_len * 2 - 1:
            while len(temp) != myData.df_len * 2 - 1:
                end = end - relativedelta(days=1)
                temp = yf.download('MCD', str(start)[:10], str(end)[:10], progress = False)
        elif len(temp) < myData.df_len * 2 - 1:
            while len(temp) != myData.df_len * 2 - 1:
                end = end + relativedelta(days=1)
                temp = yf.download('MCD', str(start)[:10], str(end)[:10], progress = False)
        myData.start = start
        myData.end = end
        df_dict, y_dict = data_yf(myData).retFreq()

        threshold = 0

        X = torch.Tensor([])
        y = []
        for k in df_dict.keys():
            _X = torch.Tensor(np.array(df_dict[k])).view(-1, 1, myData.df_len-1, myData.df_len-1)
            X = torch.cat((X, _X), 0)
            _y = [0, 1] if y_dict[k].mean() >= threshold else [1, 0]
            y.append(_y)
        y = torch.Tensor(np.array(y))

        train_split = round(len(X) * 0.7)
        X_train, X_test = X[:train_split], X[train_split:]
        y_train, y_test = y[:train_split], y[train_split:]

        X_y_set = X_train, y_train, X_test, y_test

        ls = []
        for i in [0.2, 0.5, 0.7, 0.9]:
            ls.append(run_optim(alpha=0.001, miu=i, size=myData.df_len, X_y_set=X_y_set).model_test())
        ls_ls.append(ls)
    df = pd.DataFrame(ls_ls, index = date_list, columns = [0.2, 0.5, 0.7, 0.9]) 
    print(df)   