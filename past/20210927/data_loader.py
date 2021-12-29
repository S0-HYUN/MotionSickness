from operator import getitem
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import os
from tqdm import tqdm # 진행상황에 대한 정보가 시각화
from torch.utils.tensorboard import SummaryWriter
import MS_Dataset
import filing
import setting
from sklearn.model_selection import train_test_split

class Dataset(Dataset) : # torch.utils.data.Dataset -> 데이터셋을 나타내는 추상클래스
    one_bundle = setting.one_bundle
    channel_num = setting.channel_num

    def __init__(self, path, subj_num, v_list, data_type = 'train'): # 클래스의 생성자 함수
        dataload = MS_Dataset.Load_Data(path, subj_num)
        dataload_v = MS_Dataset.Load_Data(path, v_list)
        self.data = torch.tensor(dataload.o_datalist)
        self.data_t = torch.tensor(dataload_v.o_datalist)

        one_bundle = setting.one_bundle
        channel_num = setting.channel_num

        # print("data shape", self.data.shape)        # [1247, 750, 29]
        # print("data_t shape", self.data_t.shape)    # [464, 750, 29]

        self.x = self.data[:,:,:-1].reshape(-1,one_bundle,channel_num) # [1247, 750, 28]
        self.y = self.data[:,:,-1].reshape(-1,one_bundle,1).mean(-2) # [1247, 1]
        # self.y = self.y.reshape(-1,1,1)

        # print("x shape", self.x.shape)
        # print("y shape", self.y.shape)

        self.x_v = self.data_t[:,:,:-1].reshape(-1,one_bundle,channel_num) # [464, 750, 28]
        self.y_v = self.data_t[:,:,-1].reshape(-1,one_bundle,1).mean(-2) # [464, 750, 1]  -> [464, 1, 1]
        # self.y_v = self.y_v.reshape(-1, 1, 1)
        
        # print("x_val shape", self.x_v.shape)
        # print("y_val shape", self.y_v.shape)

        x_n = self.x_v.numpy(); y_n = self.y_v.numpy()
        x_train, x_val, y_train, y_val = train_test_split(x_n, y_n, test_size=setting.test_size, shuffle=True, random_state=1004)

        # print("numpy shape", x_n.shape) # [464, 750, 28]
        
        # print("x_train shape", x_train.shape) # [623, 750, 28]
        # print("x_val", x_val.shape) # [232, 750 , 28]
        # print("y_train shape", y_train.shape) # [232, 1, 1]
        # print("y_val", y_val.shape) # [232, 1, 1]
        # print("x값은:", self.x.shape)
        # print("y값은:", self.y.shape)
 

        if data_type == 'train' :
            if subj_num == v_list :
                print("들어옴!")
                self.x = torch.tensor(x_train)
                self.y = torch.tensor(y_train)
            else :
                self.x = torch.tensor(np.vstack([self.x.numpy(), x_train])) # [1479, 750, 28]
                self.y = torch.tensor(np.vstack([self.y.numpy(), y_train])) # [1479, 1]
            # print(self.y)

        elif data_type == 'valid' :
            if subj_num == v_list :
                print("여기도 들어왔음")
                self.x = torch.tensor(x_val)
                self.y = torch.tensor(y_val)
            else :
                self.x = torch.tensor(x_val) # [232, 750, 28]
                self.y = torch.tensor(y_val) # [232, 1, 1]
            # print("y", self.y)


        elif data_type == 'test' :
            self.x = torch.tensor(self.x)
            self.y = torch.tensor(self.y)

        else: # 에러 발생
            raise

  
    def __len__(self): # len(dataset) -> 데이터셋의 크기를 리턴해야
        return len(self.x)

    def __getitem__(self, idx): # i번째 샘플을 찾는데 사용
        return self.x[idx], self.y[idx]


#---# trigger 전후 3초 데이터 뽑기 #---#
# data_df = pd.DataFrame(self.data.numpy())
# filing.use_naming(data_df)
# print("printing data: ", data_df.head)

# time_cal = (pd.DataFrame(data_df['Time']) - data_df.iloc[0][0]) % 60
# test_idx = time_cal.index[(time_cal['Time'] < 3) | (time_cal['Time'] > 57)].tolist()
# data_df = data_df.loc[test_idx,:]
# data_df.reset_index(drop = True, inplace = True) # index 재배열 0부터~

# data = torch.tensor(data_df.values)
# self.x = data[:,:-1]
# self.y = data[:,-1]
            
#o_data_t = torch.tensor(o_data)