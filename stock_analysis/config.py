import numpy as np
import pandas as pd
import os
os.chdir(os.getcwd()+'/stock_analysis')
# The csv for food stocks is downloaded from https://companiesmarketcap.com/food/largest-food-companies-by-market-cap/#google_vignette
# df = pd.read_csv(os.getcwd()+'/data/food_stock.csv')
# The csv for tech stocks is downloaded from https://companiesmarketcap.com/tech/largest-tech-companies-by-market-cap/
df = pd.read_csv(os.getcwd()+'/data/tech_stock.csv')
ticker = list(df['Symbol'])
start = '2021-09-30'
end = '2023-10-03'
df_len = 252
shift_period = list(np.arange(0, df_len + 1, 1))