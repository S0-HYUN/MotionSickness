from operator import getitem
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np

class Dataset(Dataset) :
    def __init__(self, info, path, t_list, v_list, data_type = 'train'):
        if data_type == 'train' :
            self.data = self.make_training(info, path, t_list, v_list)
            self.x = torch.tensor(self.data[0]) # [18961, 750, 28]
            self.y = torch.tensor(self.data[1]).mean(-1) # [18961]

        elif data_type == 'valid' :
            self.data_t = self.make_valid(info, path, v_list)
            self.x = torch.tensor(self.data_t[0])
            self.y = torch.tensor(self.data_t[1]).mean(-1)

        elif data_type == 'test' :
            self.data_test = self.make_test(info, path, v_list)
            self.x = torch.tensor(self.data_test[0])
            self.y = torch.tensor(self.data_test[1]).mean(-1)
        
        else :
            raise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx): 
        return self.x[idx], self.y[idx]

    def make_name(self, category_ss, test_size, class_num, expt, day, subj_num, category_tv):
        '''
        make data file name function
        input     : category_ss [Split / Single], test_size, num of class, expt, day, subj_num, category_tv [train / valid]
        output    : data file name
        '''
        if test_size :
            data_name = category_ss + str(test_size)
        else :
            data_name = category_ss
        
        data_name = data_name + \
                    "/Class" + str(class_num) + \
                    "/Expt" + str(expt) + \
                    "/day" + str(day) + \
                    "/subj" + subj_num.zfill(2) + category_tv

        return data_name

    def make_training(self, info, path, list_, list_v_):
        '''
        make training data stack
        input   : info, path, list_(train list), list_v(valid list)
        output  : train list stack
        '''
        total_list_x = []
        total_list_y = []

        for sub in list_ :            
            for d in range(2) :
                data_name = self.make_name("Single", None, info['class'], info['expt'], (d+1), str(sub), ".npz")
                o_list = np.load(path + data_name) # "subj01_day1_train.npz"
                total_list_x.append(o_list['x'])
                total_list_y.append(o_list['y'])

        # for sub in list_v_ :
        #     data_name = self.make_name("Split", info['test_size'], info['class'], info['expt'], 1, str(sub), "_train.npz")
        #     o_list = np.load(path + data_name)
        #     total_list_x.append(o_list['x'])
        #     total_list_y.append(o_list['y'])
            
        return np.vstack(total_list_x), np.vstack(total_list_y)
    
    def make_valid(self, info, path, list_):
        '''
        make valid data stack
        input   : info, path, list_(valid list)
        output  : valid list stack
        '''
        total_list_x = []
        total_list_y = []
        
        for sub in list_ :
            data_name = self.make_name("Split", info['test_size'], info['class'], info['expt'], 1, str(sub), "_val.npz")

            o_list = np.load(path + data_name)
            total_list_x.append(o_list['x'])
            total_list_y.append(o_list['y'])
        
        return np.vstack(total_list_x), np.vstack(total_list_y)

    def make_test(self, info, path, list_):
        '''
        make test data stack
        input   : info, path, list_(test list)
        output  : test list stack
        '''
        total_list_x = []
        total_list_y = []

        for sub in list_ :            
            data_name = self.make_name("Split", info['test_size'], info['class'], info['expt'], 2, str(sub), "_val.npz")

            o_list = np.load(path + data_name)
            total_list_x.append(o_list['x'])
            total_list_y.append(o_list['y'])

        return np.vstack(total_list_x), np.vstack(total_list_y)