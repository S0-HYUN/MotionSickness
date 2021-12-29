from operator import getitem
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
from utils import *

class Dataset(Dataset) :
    def __init__(self, args):
        print("\nData Loading...")
        train_list = data_preprocesesing(list(range(1,24)), args.remove_subj, [args.test_subj])
        if args.mode == "train":
            self.data = self.make_training(args, args.path, train_list)
            self.x = torch.tensor(self.data[0]) # [18961, 750, 28]
            self.y = torch.tensor(self.data[1]).mean(-1) # [18961]

            self.data_v = self.make_valid(args, [args.test_subj])

        elif args.mode == 'test' :
            self.data_test = self.make_test(args, [args.test_subj])
            self.x = torch.tensor(self.data_test[0])
            self.y = torch.tensor(self.data_test[1]).mean(-1)
        
        else :
            raise TypeError

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

    def make_training(self, args, list_):
        '''
        make training data stack
        input   : args, path, list_(train list), list_v(valid list)
        output  : train list stack

        all of Day1 + (1 - test_ratio) * Day2  
        '''
        total_list_x = []
        total_list_y = []

        for sub in list_: ################################################### 이게 최선인가여.. 맘에 안들어.
            data_name = self.make_name("Single", None, args.class_num, args.expt, 1, str(sub), ".npz")
            o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

            data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 2, str(sub), "_train.npz")
            o_list = np.load(args.path + data_name) # "subj01_train.npz"
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

            data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 2, str(sub), "_val.npz")
            o_list = np.load(args.path + data_name) # "subj01_val.npz"
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # for sub in list_t_ :
        #     data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(sub), "_train.npz")
        #     o_list = np.load(path + data_name)
        #     total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
            
        return np.vstack(total_list_x), np.vstack(total_list_y)
    
    def make_valid(self, args, list_):
        '''
        make valid data stack
        input   : args, path, list_(valid list)
        output  : valid list stack
        '''
        total_list_x = []
        total_list_y = []
        
        for sub in list_ :
            data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(sub), "_train.npz")
            o_list = np.load(args.path + data_name)
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        
        return np.vstack(total_list_x), np.vstack(total_list_y)

    def make_test(self, args, list_):
        '''
        make test data stack
        input   : args, path, list_(test list)
        output  : test list stack
        '''
        total_list_x = []
        total_list_y = []

        for sub in list_ :            
            data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 2, str(sub), "_val.npz")

            o_list = np.load(args.path + data_name)
            total_list_x.append(o_list['x'])
            total_list_y.append(o_list['y'])

        return np.vstack(total_list_x), np.vstack(total_list_y)