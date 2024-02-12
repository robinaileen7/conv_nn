import os
import sys
sys.path.append(os.getcwd())
from data_prep import df_dict 
import matplotlib.pyplot as plt

regenerate_pics = True
threshold = 0
good = []
bad = []
for k in df_dict.keys():
    if df_dict[k]['Ret_252'].mean() > threshold:
        good.append(k)
    else:
        bad.append(k)

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

if regenerate_pics:
    save_pics(key_ls=good, dir_name='good')
    save_pics(key_ls=bad, dir_name='bad')