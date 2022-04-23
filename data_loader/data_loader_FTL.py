from operator import getitem
from numpy.core.shape_base import vstack
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *

class Dataset() :
    def __init__(self, args, phase):
        print(f"Data Loading... ({phase})")
        train_list = data_preprocesesing(list(range(1,24)), [args.test_subj], args.remove_subj)
        self.phase = phase
        
        if args.mode == "train":
            #---# train / pool / valid #---#
            if self.phase == "train":
                self.data = self.make_training(args, train_list)
                # self.data = self.make_training(args, [args.test_subj])
   
            elif self.phase == "valid":    
                self.data = self.make_valid(args, [args.test_subj])
                # self.data = self.make_valid(args, train_list)
            
            elif self.phase == "test":
                self.data = self.make_test(args, [args.test_subj])
            
        elif args.mode == "test":
            self.data = self.make_test(args, [args.test_subj])

        else :
            raise TypeError

        data_reshape = self.data[0].reshape(self.data[0].shape[0], self.data[0].shape[2], self.data[0].shape[1])
        # self.x = torch.tensor(self.data[0])
        self.x = []
        for i in range(self.data[0].shape[0]):
            dd = data_reshape[i]
            ss = np.cov(dd) / 749
            self.x.append(ss)
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.data[1]).mean(-1)
        self.subj = torch.tensor(self.data[2])

        # self.y = torch.tensor(self.data[1])
        # self.y = self.y.reshape(self.y.shape[1])

        print("sample shape : ", self.x.shape)
        self.in_weights = make_weights_for_balanced_classes(self.y)

        # self.x = torch.sum(self.x, axis=1)

        # self.x = self.x.cpu().numpy()
        # self.y = self.y.cpu().numpy()
        
        #---# check min, max #---#
        # for i in range(self.x.shape[0]):
        #     print(torch.amax(self.x[i], dim=0), torch.amin(self.x[i], dim=0))
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx): 
        return self.x[idx], self.y[idx], self.subj[idx]

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
        total_list_x = []; total_list_y = []; total_list_subj = []

        for sub in list_ :            
            for expt in [1,2] :
                data_name = self.make_name("Single", None, args.class_num, expt, 1, str(sub), ".npz")
                o_list = np.load(args.path + data_name)
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

                data_name = self.make_name("Single", None, args.class_num, expt, 2, str(sub), ".npz")
                o_list = np.load(args.path + data_name)
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))
       

        return np.vstack(total_list_x), np.vstack(total_list_y), total_list_subj
    
    def make_valid(self, args, list_):
        '''
        make valid data stack
        input   : args, path, list_(valid list)
        output  : valid list stack
        '''
        total_list_x = []; total_list_y = []; total_list_subj = []

        for sub in list_ :            
            for expt in [1,2] :
                data_name = self.make_name("Single", None, args.class_num, expt, 1, str(sub), ".npz")
                o_list = np.load(args.path + data_name)
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

        return np.vstack(total_list_x), np.vstack(total_list_y), total_list_subj

    def make_test(self, args, list_):
        '''
        make test data stack
        input   : args, path, list_(test list)
        output  : test list stack
        '''
        total_list_x = []; total_list_y = []; total_list_subj = []

        for sub in list_ :            
            for expt in [1,2] :
                data_name = self.make_name("Single", None, args.class_num, expt, 2, str(sub), ".npz")
                o_list = np.load(args.path + data_name)
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))
        
        return np.vstack(total_list_x), np.vstack(total_list_y), total_list_subj

from collections import Counter
# get weights (use the number of samples)
def make_weights_for_balanced_classes(dataset):  # y값 class에 따라
    counts = Counter()
    classes = []
    for y in dataset:
        y = int(y.item()) # class에 접근
        counts[y] += 1 # count each class samples
        classes.append(y) 
    n_classes = len(counts)

    weight_per_class = {}
    for y in counts: # the key of counts
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[y]

    return weights