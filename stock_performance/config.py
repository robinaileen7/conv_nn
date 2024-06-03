import numpy as np
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.getcwd()+'/stock_performance')
# https://companiesmarketcap.com

#df = pd.read_csv(os.getcwd()+'/data/mkt_cap.csv')
#ticker = list(df['Symbol'])[:1000]

df = pd.read_excel(os.getcwd()+ '/data/stock_vol.xlsx')
ticker = list(df['ticker'])
start = '2022-03-01'
end = '2024-03-01'
df_len = 252
shift_period = list(np.arange(0, df_len, 1))