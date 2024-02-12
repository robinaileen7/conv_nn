import cv2
import os
os.chdir(os.getcwd()+'/stock_analysis')
from os.path import isfile, join
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split

class def_class:
    def __init__(self):
        self.img_dir = os.getcwd() + '\\pics\\'
        self.resize_ind = True
        self.size = 100
        self.colored = False
        self.train_test_split = 0.25
        # self.label = {'food': 0, 'tech': 1}
        self.label = {'good': 0, 'bad': 1}
    def load_pics(self):
        print(self.label)
        pic_dict = {}
        pic_key = []
        for i in self.label:
            path = self.img_dir + i + '\\'
            file_name = [g for g in os.listdir(path) if isfile(join(path, g))]
            for j in file_name:
                pic_key.append(j.split('.jpg')[0])
                if self.colored:
                    img = cv2.imread(path + j)
                else:
                    img = cv2.imread(path + j, cv2.IMREAD_GRAYSCALE)
                if self.resize_ind:
                    img = cv2.resize(img, (self.size, self.size))
                ind = np.zeros(len(self.label))
                ind[self.label[i]] = 1
                pic_dict[j.split('.jpg')[0]] = [ind, np.array(img)]
        X = np.array([k[1] for k in list(pic_dict.values())])
        y = np.array([k[0] for k in list(pic_dict.values())])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.train_test_split, random_state=10)
        if self.colored:
            X_train = torch.Tensor(X_train).view(-1, 3, self.size, self.size)
        else:
            X_train = torch.Tensor(X_train).view(-1, 1, self.size, self.size)
        y_train = torch.Tensor(y_train)
        random.seed(0)
        return random.sample(pic_key, len(pic_key)), pic_dict, X_train, X_test, y_train, y_test
    
pic_key, pic_dict, X_train, X_test, y_train, y_test = def_class().load_pics()
#print(X_test, y_test)