import os
import sys
sys.path.append(os.getcwd())
from data_prep import df_dict 
import matplotlib.pyplot as plt
import numpy as np
from config import df_len
import torch

regenerate_pics = False
threshold = 0.2
X = torch.Tensor([])
y = []
for k in df_dict.keys():
    _X = torch.Tensor(np.array(df_dict[k])).view(-1, 1, df_len, df_len)
    X = torch.cat((X, _X), 0)
    _y = [0, 1] if np.array(df_dict[k])[:, -1].mean() >= threshold else [1, 0]
    y.append(_y)
y = torch.Tensor(np.array(y))

train_split = round(len(X) * 0.75)
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

def save_pics(key_ls, dir_name):
    path = os.getcwd()+'/pics/' + str(dir_name) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in key_ls:
        plt.imshow(df_dict[i].to_numpy())
        plt.axis('off')
        try:
            plt.savefig(path+ str(i) + ".jpg", bbox_inches='tight', pad_inches=0)
        except ValueError as e:
            print(i + ' cannot be converted to an image')
if __name__ == "__main__":
    if regenerate_pics:
        good = []
        bad = []
        for k in df_dict.keys():
            if df_dict[k]['Ret_252'].mean() > threshold:
                good.append(k)
            else:
                bad.append(k)
        save_pics(key_ls=good, dir_name='good')
        save_pics(key_ls=bad, dir_name='bad')