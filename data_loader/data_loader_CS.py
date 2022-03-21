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
        train_list = data_preprocesesing(list(range(1,16)), [args.test_subj])
        # train_list = data_preprocesesing(list(range(1,10)), args.remove_subj, [args.test_subj])
        self.phase = phase
        
        if args.mode == "train":
            #---# train / pool / valid #---#

            if self.phase == "train":
                self.data = self.make_training(args, train_list)
                # self.data = self.make_training(args, [args.test_subj])

            # elif self.phase == "pool":
            #     self.data = self.make_pool(args, train_list)
            #     # self.data = self.make_pool(args, [args.test_subj])
   
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
        self.y = torch.tensor(self.data[1]).flatten()

        # self.y = torch.tensor(self.data[1]).float().mean(-1)

        # self.y = torch.tensor(self.data[1])
        # self.y = self.y.reshape(self.y.shape[1])

        print(self.x.shape, self.y.shape)
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
        return self.x[idx], self.y[idx]

    def make_name(self, category_ss, test_size, class_num, video_num, subj_num, category_tv):
        '''
        make data file name function
        input     : category_ss [Split / Single], test_size, num of class, video_num, subj_num, category_tv [train / valid]
        output    : data file name
        '''
        if test_size :
            data_name = category_ss + str(test_size)
        else :
            data_name = category_ss
        
        data_name = data_name + \
                    "/Class" + str(class_num) + \
                    "/V" + str(video_num) + \
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

        for sub in list_:
            for v_n in range(1,args.video_num+1) :
                data_name = self.make_name("Split", args.test_size, args.class_num, v_n, str(sub), "_train.npz")
                try : o_list = np.load(args.path + data_name)
                except : continue
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        
        # data_name = self.make_name("Split", args.test_size, args.class_num, v_n, str(args.test_subj), "_train.npz")
        # try : o_list = np.load(args.path + data_name)
        # except : pass
        # total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        return np.vstack(total_list_x), np.vstack(total_list_y)

    def make_valid(self, args, list_):
        '''
        make valid data stack
        input   : args, path, list_(valid list)
        output  : valid list stack
        '''
        total_list_x = []
        total_list_y = []
        
        for sub in list_:
            for v_n in range(1,args.video_num+1) :
                data_name = self.make_name("Split", args.test_size, args.class_num, v_n, str(sub), "_val.npz")
                try : o_list = np.load(args.path + data_name)
                except : continue
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        ata_name = self.make_name("Split", args.test_size, args.class_num, v_n, str(args.test_subj), "_val.npz")
        try : o_list = np.load(args.path + data_name)
        except : pass
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

        for sub in list_:
            for v_n in range(1,args.video_num+1) :
                # data_name = self.make_name("Single", args.test_size, args.class_num, v_n, str(sub), ".npz")
                data_name = self.make_name("Split", args.test_size, args.class_num, v_n, str(sub), "_train.npz")
                try : o_list = np.load(args.path + data_name)
                except : continue
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        return np.vstack(total_list_x), np.vstack(total_list_y)

    # 아직 수정 안함
    def make_training_da(self, args, list_):
        total_list_x = []; total_list_y = []
        for sub in list_ :    
            data_name = self.make_name("Single", None, args.class_num, args.expt, 1, str(sub), ".npz")        
            # data_name = self.make_name("Split", 0.5, args.class_num, args.expt, 1, str(sub), "_val.npz")
            o_list = np.load(args.path + data_name)
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        return np.vstack(total_list_x), np.vstack(total_list_y)

    def make_test_da(self, args, list_):
        total_list_x = []; total_list_y = []
        for sub in list_ :            
            data_name = self.make_name("Single", None, args.class_num, args.expt, 2, str(sub), ".npz")
            o_list = np.load(args.path + data_name)
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        return np.vstack(total_list_x), np.vstack(total_list_y)


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