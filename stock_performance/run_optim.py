import torch.optim as optim
import torch
dtype = torch.float
device = torch.device("cuda:0") # Uncommon this to run on GPU
# device = torch.device("cpu") # Uncommon this to run on CPU
from run_nn import Net, model
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
import os
import sys
sys.path.append(os.getcwd())

save_model = False
load_model = False
equal_weight = False

class run_optim:
    def __init__(self, alpha, miu, size, X_y_set, ret_test, k_test, weight_test):
        self.optimizer = optim.SGD(model.parameters(), lr=alpha, momentum=miu, weight_decay=0.2)
        self.colored = False
        self.size = size
        self.X_train = X_y_set[0]
        self.y_train = X_y_set[1]
        self.X_test = X_y_set[2]
        self.y_test = X_y_set[3]
        self.ret_test = ret_test
        self.k_test = k_test
        self.weight_test = weight_test
        self.batch_size = round(len(self.X_train)/2)
        self.miu = miu
        self.trading_num = 100

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

        epochs = 1
        for j in range(epochs):
            for i in range(0, len(X_train)-batch_size):
                optimizer.zero_grad()
                loss = run_optim.loss_function('MSE')(model(X_train[i:i+batch_size]), y_train[i:i+batch_size])
                loss.backward()
                optimizer.step()

        print("Loss for training set is")
        print(loss)

        if save_model:
            file_path = os.getcwd() + '/pretrained/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            #torch.save(model.state_dict(), file_path + str(self.miu) + '.pth')
                
            checkpoint = { 
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'epochs': epochs}
            torch.save(checkpoint, file_path + str(self.miu) + '.pth')
        
        #print("Model's state_dict:")
        #for param_tensor in model.state_dict():
            #print(param_tensor, "\t", model.state_dict()[param_tensor])

        #print("Optimizer's state_dict:")
        #for var_name in optimizer.state_dict():
            #print(var_name, "\t", optimizer.state_dict()[var_name])

    def model_test(self):
        if not load_model:
            run_optim.model_train(self)
        else:
            file_path = os.getcwd() + '/pretrained/' + str(self.miu) + '.pth'
        
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            
        size = self.size
        X_test = self.X_test
        y_test = self.y_test
        ret_test = self.ret_test
        k_test = self.k_test
        weight_test = self.weight_test

        correct = 0
        ret = 0
        buy = []
        sell = []
        buy_ad_hoc = []
        with torch.no_grad():
            for i in range(0, len(X_test)):
                if self.colored:
                    y_hat = model(torch.Tensor(X_test[i]).view(-1, 3, size-1, size-1))
                else:
                    y_hat = model(torch.Tensor(X_test[i]).view(-1, 1, size-1, size-1))
                y = torch.Tensor(y_test[i])
                if torch.argmax(y_hat) == 1:
                    if equal_weight:
                        ret += ret_test[i]/len(X_test)
                    else:
                        ret += ret_test[i]*weight_test[i]
                    buy.append(k_test[i])
                else:
                    if equal_weight:
                        ret -= ret_test[i]/len(X_test)
                    else:
                        ret -= ret_test[i]*weight_test[i]
                    sell.append(k_test[i])
                if torch.argmax(y_hat) == torch.argmax(y):
                    correct += 1
                    if torch.argmax(y_hat) == 1:
                        buy_ad_hoc.append(k_test[i])
        print("The accuracy is")
        print(correct/len(y_test))
        print("The return is")
        print(ret)
        print("The assets to buy:")
        print(buy)
        print("The assets to sell:")
        print(sell)
        print("Ad Hoc to buy assets:")
        print(buy_ad_hoc)
        
        return correct/len(y_test), ret, buy, sell, buy_ad_hoc

class myStruct():
    pass

myData = myStruct()
myData.ticker = config.ticker
myData.period = config.shift_period
myData.df_len = config.df_len

ret_stru = 'Daily' # 'Weekly'

if __name__ == "__main__":
    date_list = pd.date_range(datetime.datetime(2022, 3, 1), datetime.datetime(2022, 3, 29), freq='D')
    # date_list = pd.date_range(datetime.datetime(2022, 3, 1), datetime.datetime(2022, 3, 29), freq='W-TUE')
    ls_ls = []
    ret_ls = []
    for j in date_list:
        print(j)
        start = j
        end = start + relativedelta(years=2)
        temp = yf.download('MSFT', str(start)[:10], str(end)[:10], progress = False)
        if len(temp) > myData.df_len * 2 - 1:
            while len(temp) != myData.df_len * 2 - 1:
                end = end - relativedelta(days=1)
                temp = yf.download('MSFT', str(start)[:10], str(end)[:10], progress = False)
        elif len(temp) < myData.df_len * 2 - 1:
            while len(temp) != myData.df_len * 2 - 1:
                end = end + relativedelta(days=1)
                temp = yf.download('MSFT', str(start)[:10], str(end)[:10], progress = False)
        myData.start = start
        myData.end = end
        df_dict, y_dict = data_yf(myData).retFreq()

        threshold = 0

        X = torch.Tensor([])
        y = []
        ret = []
        vol = []
        for k in df_dict.keys():
            _X = torch.Tensor(np.array(df_dict[k])).view(-1, 1, myData.df_len-1, myData.df_len-1)
            X = torch.cat((X, _X), 0)
            if ret_stru == 'Daily':
                _y = [0, 1] if y_dict[k]['Ret_1'] >= threshold else [1, 0]
                y.append(_y)
                ret.append(y_dict[k]['Ret_1'])
                vol.append(df_dict[k]['Ret_1'].std())
            elif ret_stru == 'Weekly':
                _y = [0, 1] if y_dict[k]['Ret_5'] >= threshold else [1, 0]
                y.append(_y)
                ret.append(y_dict[k]['Ret_5'])
                vol.append(df_dict[k]['Ret_5'].std())
        y = torch.Tensor(np.array(y))
        train_split = round(len(X) * 0.7)
        X_train, X_test = X[:train_split], X[train_split:]
        y_train, y_test = y[:train_split], y[train_split:]

        X_y_set = X_train, y_train, X_test, y_test
        ret_test = ret[train_split:]
        k_test = list(df_dict.keys())[train_split:]
        vol_test = vol[train_split:]
        reversed_vol_test = [1/x for x in vol_test]
        reversed_weight_test = [x/np.sum(reversed_vol_test) for x in reversed_vol_test]
        ls = []
        _ls = []
        momentum_ls = [0.5, 0.9]
        for i in momentum_ls:
            set_seed = 0
            np.random.seed(set_seed)
            torch.manual_seed(set_seed)
            torch.cuda.manual_seed(set_seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(torch.cuda.is_available())
            acc, _ret, to_buy, to_sell, to_buy_ad = run_optim(alpha=0.001, miu=i, size=myData.df_len, X_y_set=X_y_set, ret_test=ret_test, k_test=k_test, weight_test=reversed_weight_test).model_test()
            ls.append(acc)
            _ls.append(_ret)
        ls_ls.append(ls)
        ret_ls.append(_ls)
    df = pd.DataFrame(ls_ls, index = date_list, columns = momentum_ls) 
    df_ret = pd.DataFrame(ret_ls, index = date_list, columns = momentum_ls) 
    print(df)  
    print(df_ret) 