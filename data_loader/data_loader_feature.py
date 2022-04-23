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
from utils import *

class Dataset(Dataset) :
    def __init__(self, args, phase):
        print(f"Data Loading... ({phase})")
        train_list = data_preprocesesing(list(range(1,24)), [args.test_subj], args.remove_subj)
        # train_list = data_preprocesesing(list(range(1,10)), args.remove_subj, [args.test_subj])
        self.phase = phase
        
        if args.mode == "train":
            #---# train / pool / valid #---#

            if self.phase == "train":
                self.data = self.make_training(args, train_list)
                # self.data = self.make_training(args, [args.test_subj])

            elif self.phase == "valid":    
                # self.data = self.make_valid(args, [args.test_subj])
                self.data = self.make_valid(args, train_list)
            
            elif self.phase == "test":
                self.data = self.make_test(args, [args.test_subj])
            
            elif self.phase == "DA":
                self.data = self.make_training_da(args, [args.test_subj])
            
        elif args.mode == "test":
            self.data = self.make_test(args, [args.test_subj])

        elif self.phase == "DA_test":
                self.data = self.make_test_da(args, [args.test_subj]) 
        else :
            raise TypeError

        self.x = torch.tensor(self.data[0])
        self.y = torch.tensor(self.data[1])
        self.subj = torch.tensor(self.data[2])

        # self.y = torch.tensor(self.data[1])
        # self.y = self.y.reshape(self.y.shape[1])

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

    def make_training(self, args, list_):
        total_list_feature = []; total_list_y = []; total_list_subj = []

        for sub in list_:
            data_name = "original_" + "subj" + str(sub).zfill(2) + ".npz"
            o_list = np.load(args.path + data_name)
            total_list_feature.append(o_list['arr_0']); total_list_y.extend(o_list['arr_1']); 
            total_list_subj.extend(np.repeat(sub, o_list['arr_0'].shape[0]))

        return np.vstack(total_list_feature), total_list_y, total_list_subj
  
    def make_valid(self, args, list_):
        total_list_x = []
        total_list_y = []

        for sub in list_:  
            data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(sub), "_val.npz")
            o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

            data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 2, str(sub), "_val.npz")
            o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        return np.vstack(total_list_x), np.vstack(total_list_y)

    def make_test(self, args, list_):
        total_list_feature = []; total_list_y = []; total_list_subj = []

        for sub in list_:
            data_name = "original_" + "subj" + str(sub).zfill(2) + ".npz"
            o_list = np.load(args.path + data_name)
            total_list_feature.append(o_list['arr_0']); total_list_y.extend(o_list['arr_1']); 
            total_list_subj.extend(np.repeat(sub, o_list['arr_0'].shape[0]))

        return np.vstack(total_list_feature), total_list_y, total_list_subj


class ActiveDataset(Dataset) :
    def __init__(self, args, x_, y_):
        print(f"Active Learning Data Loading...")
        
        self.x = x_
        self.y = y_
        
        # self.x = torch.tensor(self.data[0])
        # self.y = torch.tensor(self.data[1]).mean(-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx): 
        return self.x[idx], self.y[idx]

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