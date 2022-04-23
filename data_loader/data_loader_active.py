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
    def __init__(self, args, phase, rest = False):
        print(f"Data Loading... ({phase})")
        train_list = data_preprocesesing(list(range(1,24)), [args.test_subj], args.remove_subj)

        # train_list = data_preprocesesing(list(range(1,10)), args.remove_subj, [args.test_subj])
        self.phase = phase
        self.rest = rest   
        self.args = args
        
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
        self.y = torch.tensor(self.data[1]).mean(-1)
        self.subj = torch.tensor(self.data[2])

        # self.y = torch.tensor(self.data[1])
        # self.y = self.y.reshape(self.y.shape[1])

        print("sample shape : ", self.x.shape)
        if not self.rest :
            self.bins = BucketingSampler(self.y, self.args.batch_size)

        # self.in_weights = make_weights_for_balanced_classes(self.y)

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

        # for sub in list_: ################################################### 이게 최선인가여.. 맘에 안들어.
        #     for d in range(1,3): 
        #         data_name = self.make_name("Single", None, args.class_num, args.expt, d, str(sub), ".npz")
        #         o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
        #         total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(args.test_subj), "_train.npz")
        # o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
        # total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # 이걸로 계속 가자
        if self.rest : path = args.rest_path; print("--rest--")
        else : path = args.path; print("--MS--")

        for sub in list_:   
            for expt in [1, 2]:
                data_name = self.make_name("Split", args.test_size, args.class_num, expt, 1, str(sub), "_train.npz")
                o_list = np.load(path + data_name) # "subj01_day1_train.npz"
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

                data_name = self.make_name("Split", args.test_size, args.class_num, expt, 2, str(sub), "_train.npz")
                o_list = np.load(path + data_name) # "subj01_day1_train.npz"
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

                del o_list
               

        data_name = self.make_name("Split", args.test_size, args.class_num, 1, 1, str(args.test_subj), "_train.npz")
        o_list = np.load(path + data_name)
        total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

        data_name = self.make_name("Split", args.test_size, args.class_num, 2, 1, str(args.test_subj), "_train.npz")
        o_list = np.load(path + data_name)
        total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

        del o_list

        return np.vstack(total_list_x), np.vstack(total_list_y), total_list_subj

    def make_valid(self, args, list_):
        '''
        make valid data stack
        input   : args, path, list_(valid list)
        output  : valid list stack
        '''
        total_list_x = []; total_list_y = []; total_list_subj = []

        # for sub in list_ :            
        #     # data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(sub), "_val.npz")
        #     data_name = self.make_name("Single", None, args.class_num, args.expt, 1, str(sub), ".npz")
        #     o_list = np.load(args.path + data_name)
        #     total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        #     data_name = self.make_name("Single", None, args.class_num, args.expt, 2, str(sub), ".npz")
        #     o_list = np.load(args.path + data_name)
        #     total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        if self.rest : path = args.rest_path
        else : path = args.path
        for sub in list_:  
            for expt in [1,2]:
                data_name = self.make_name("Split", args.test_size, args.class_num, expt, 1, str(sub), "_val.npz")
                o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

                data_name = self.make_name("Split", args.test_size, args.class_num, expt, 2, str(sub), "_val.npz")
                o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

                del o_list

        data_name = self.make_name("Split", args.test_size, args.class_num, 1, 1, str(args.test_subj), "_val.npz")
        o_list = np.load(args.path + data_name)
        total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

        data_name = self.make_name("Split", args.test_size, args.class_num, 2, 1, str(args.test_subj), "_val.npz")
        o_list = np.load(args.path + data_name)
        total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))

        # for sub in list_ :
        #     data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 2, str(sub), "_val.npz")
        #     o_list = np.load(args.path + data_name)
        #     total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
   
        # for sub in list_: ################################################### 이게 최선인가여.. 맘에 안들어.
            # for d in range(1,3): 
                # data_name = self.make_name("Single", None, args.class_num, args.expt, d, str(sub), ".npz")
                # data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, d, str(sub), "_val.npz")
                # o_list = np.load(args.path + data_name)
                # total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # for sub in list_ : 
        #     data_name = self.make_name("Single", None, args.class_num, args.expt, 1, str(sub), ".npz")
        #     o_list = np.load(args.path + data_name)
        #     total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        
        return np.vstack(total_list_x), np.vstack(total_list_y), total_list_subj

    def make_test(self, args, list_):
        '''
        make test data stack
        input   : args, path, list_(test list)
        output  : test list stack
        '''
        total_list_x = []; total_list_y = []; total_list_subj = []
        if self.rest : path = args.rest_path
        else : path = args.path
        for sub in list_ :            
            # data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(sub), "_val.npz")
            # data_name = self.make_name("Single", None, args.class_num, args.expt, 1, str(sub), ".npz")
            # o_list = np.load(args.path + data_name)
            # total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
            for expt in [1,2] :
                data_name = self.make_name("Single", None, args.class_num, expt, 2, str(sub), ".npz")
                o_list = np.load(args.path + data_name)
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y']); total_list_subj.extend(np.repeat(sub, o_list['x'].shape[0]))
        
        return np.vstack(total_list_x), np.vstack(total_list_y), total_list_subj

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

from torch.utils.data import Sampler
class BucketingSampler(Sampler):
    def __init__(self, dataset, batch_size=1):
        super(BucketingSampler, self).__init__(dataset)
        self.dataset = dataset
        self.bins = [] # for 이중 리스트
        class0_idx_bool = self.dataset == 0; class1_idx_bool = self.dataset == 1; class2_idx_bool = self.dataset == 2
        
        if batch_size % 3 == 1 : 
            for i in range(int(len(dataset) / batch_size)) :
                bin = []
                # random_class = np.random.randint(args.class_num)
                self.count = int((batch_size-1) / 3)
                
                choose_0class = np.random.choice(torch.where(class0_idx_bool == True)[0], self.count)
                choose_1class = np.random.choice(torch.where(class1_idx_bool == True)[0], self.count+1) # 원래 남는 클래스하나는 랜덤으로 만들어야 하는 것이 맞지만, 우선 1 클래스가 많으니 이렇게 배치
                choose_2class = np.random.choice(torch.where(class2_idx_bool == True)[0], self.count)
                bin.extend(choose_0class); bin.extend(choose_1class); bin.extend(choose_2class)
                self.bins.append(bin)
                

        elif batch_size % 3 == 2 :
            for i in range(int(len(dataset) / batch_size)) :
                bin = []

                self.count = int((batch_size-2) / 3)
                class0_idx_bool = self.dataset == 0; class1_idx_bool = self.dataset == 1; class2_idx_bool = self.dataset == 2
                choose_0class = np.random.choice(torch.where(class0_idx_bool == True)[0], self.count+1)
                choose_1class = np.random.choice(torch.where(class1_idx_bool == True)[0], self.count+1)
                choose_2class = np.random.choice(torch.where(class2_idx_bool == True)[0], self.count)
                bin.extend(choose_0class); bin.extend(choose_1class); bin.extend(choose_2class)
                self.bins.append(bin)

    def __iter__(self):
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)