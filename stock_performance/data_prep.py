import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt
import config
import warnings
warnings.filterwarnings("ignore")

regenerate_pics = False
export_tk_vol = False

class myStruct():
    pass

myData = myStruct()
myData.ticker = config.ticker
myData.start = config.start
myData.end = config.end
myData.period = config.shift_period
myData.df_len = config.df_len

class data_yf:

    def __init__(self, myData):
        self.ticker = myData.ticker
        self.start = myData.start
        self.end = myData.end
        self.period = myData.period
        self.df_len = myData.df_len

    def extract(self, tk):
        return yf.download(tk, self.start, self.end, progress = False)
    
    def retFreq(self):
        df_dict = {}
        y_dict = {}
        for i in self.ticker:
            try:
                temp = self.extract(i)
                df_prep = temp['Adj Close'].shift(periods=self.period)
                df = pd.DataFrame()
                for j in self.period[1:]:
                    try:
                        df['Ret_' + str(j)] = (df_prep['Adj Close_0'] - df_prep['Adj Close_' + str(j)])/df_prep['Adj Close_' + str(j)]/j
                    except Exception:
                        pass
                if df.dropna().shape==(self.df_len, self.df_len-1):
                    df_dict[i] = df.dropna().iloc[:-1, :]
                    y_dict[i] = df.dropna().iloc[-1, :]
            except:
                pass
        return df_dict, y_dict

    
if __name__ == "__main__":
    print(data_yf(myData).extract('MCD'))
    df_dict, y_dict = data_yf(myData).retFreq()
    print(np.array(df_dict['MCD']))
    #print(y_dict['MCD'])
    #print(y_dict['MCD']['Ret_1'])
    if export_tk_vol:
        path = os.getcwd()+'/data/'
        vol = []
        for i in df_dict.keys():
            vol.append(df_dict[i]['Ret_1'].std())
        df = pd.DataFrame({'ticker': list(df_dict.keys()), 'vol': vol}) 
        df.to_excel(path + 'stock_vol.xlsx')
    if regenerate_pics:
        path = os.getcwd()+'/pics/'
        if not os.path.exists(path):
            os.makedirs(path)
        for x, y in df_dict.items():
            plt.imshow(df_dict[x].to_numpy())
            plt.axis('off')
            try:
                plt.savefig(path+ str(x) + ".jpg", bbox_inches='tight', pad_inches=0)
            except ValueError as e:
                print(x + ' cannot be converted to an image')
