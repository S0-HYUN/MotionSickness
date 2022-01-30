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
        # train_list = data_preprocesesing(list(range(1,24)), args.remove_subj, [args.test_subj])
        train_list = data_preprocesesing(list(range(1,10)), args.remove_subj, [args.test_subj])
        self.phase = phase
        
        if args.mode == "train":
            #---# train / pool / valid #---#

            if self.phase == "train":
                # self.data = self.make_training(args, train_list)
                self.data = self.make_training(args, [args.test_subj])

            elif self.phase == "pool":
                self.data = self.make_pool(args, train_list)
                # self.data = self.make_pool(args, [args.test_subj])
   
            elif self.phase == "valid":    
                self.data = self.make_valid(args, [args.test_subj])
            
            elif self.phase == "test":
                self.data = self.make_test(args, [args.test_subj])

        elif args.mode == "test":
            self.data = self.make_test(args, [args.test_subj])
                
        else :
            raise TypeError

        self.x = torch.tensor(self.data[0])
        self.y = torch.tensor(self.data[1]).mean(-1)

        # self.y = torch.tensor(self.data[1])
        # self.y = self.y.reshape(self.y.shape[1])
        
        xx = self.x.reshape(-1, args.channel_num)

        from scipy import signal
        fs = 250 # sampling frequency of the x time series
        wlen = 64
        hop = 14
        nfft = 512
        # f, t, Zxx = signal.stft(self.x, fs, nperseg=1000)
        # f : array of sample frequencies
        # t : array of segment times
        # Zxx : stft of x
        channel_data = []
        for i in range(args.channel_num) :
            channel_data.append(xx[:,i])            
        # print("===", channel_data[26].shape) #[897,750]
        
        # f, t, Zxx = signal.stft(channel_data[0], fs=fs, window='hamming', noverlap=wlen-hop, nfft=nfft, boundary=None, padded=False)
        # f, t, Zxx = signal.stft(channel_data[0])
        #이거 # f, t, Zxx = signal.stft(channel_data[0], fs=fs, nperseg=wlen, window='hamming', noverlap=wlen-hop)
        # print(f.shape)
        # print(t.shape)
        # print(Zxx.shape)

        # temp = channel_data[0].reshape(-1) #[672750]
        # print(temp.shape)
        
        import matplotlib.pyplot as plt
        
        # f, t, Sxx = signal.spectrogram(channel_data[0], fs=fs, nperseg=wlen, window='hamming', noverlap=wlen-hop)
        for i in range(args.channel_num) : #args.channel_num
            plt.clf()
            channel_data[i] = channel_data[i] * (1e+1)
            f, t, Sxx = signal.spectrogram(channel_data[i], fs=fs, window='hamming', noverlap=wlen-hop)
            # print(Sxx.shape); print(f.shape); print(t.shape)
            plt.pcolormesh(t, f, Sxx, shading='gouraud') 
            plt.colorbar()
            plt.clim(0,10)
            plt.ylabel('Frequency'); plt.xlabel('Time')
            plt.savefig(f'channel{str(i+1)}.png')
        
        # print("======================================")
        # for i in range(Zxx.shape[0]) :
        #     f2, t2, Sxx2 = signal.spectrogram(channel_data[0], fs=fs, nperseg=wlen, window='hamming', noverlap=wlen-hop)
        #     print("++", Sxx2.shape)
        #     print(t2.shape)
        #     print(f2.shape)
        #     print(Sxx2[i,:,:].shape)
        #     print(Sxx2)
        #     plt.pcolormesh(t2, f2, Sxx2[i,:,:], shading='gouraud')
        #     # plt.specgram(Sxx[0,:,:])
        #     plt.ylabel('Frequency')
        #     plt.xlabel('Time')
        #     plt.ylim(0,30)
        #     plt.savefig(f'222{str(i)}.png')

        #     if i == 10:
        #         break

        raise
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
            for d in range(1,3): 
                data_name = self.make_name("Single", None, args.class_num, args.expt, d, str(sub), ".npz")
                o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(args.test_subj), "_train.npz")
        # o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
        # total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # for sub in list_: ###################################################
        #     for d in range(1,3): 
        #         data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, d, str(sub), "_val.npz")
        #         o_list = np.load(args.path + data_name)
        #         total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        return np.vstack(total_list_x), np.vstack(total_list_y)
    
    def make_pool(self, args, list_):
        '''
        make training pool data stack
        input   : args, path, list_(train list), list_v(valid list)
        output  : train list stack

        50% of target day1 data 
        '''
        total_list_x = []
        total_list_y = []
    
        data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(args.test_subj), "_train.npz")
        o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
        total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
    
        
        # for sub in list_: ################################################### 이게 최선인가여.. 맘에 안들어.
        #     for d in range(2,3): #(1,3)
        #         data_name = self.make_name("Single", None, args.class_num, args.expt, d, str(sub), ".npz")
        #         o_list = np.load(args.path + data_name) # "subj01_day1_train.npz"
        #         total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        for sub in list_: ################################################### 
            for d in range(1,3): 
                data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, d, str(sub), "_train.npz")
                o_list = np.load(args.path + data_name)
                total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        

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

   
        # for sub in list_: ################################################### 이게 최선인가여.. 맘에 안들어.
        #     for d in range(1,3): 
        #         data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, d, str(sub), "_val.npz")
        #         o_list = np.load(args.path + data_name)
        #         total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(args.test_subj), "_val.npz")
        # o_list = np.load(args.path + data_name)
        # total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        
        # for sub in list_ : 
        #     data_name = self.make_name("Single", None, args.class_num, args.expt, 1, str(sub), ".npz")
        #     o_list = np.load(args.path + data_name)
        #     total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])
        
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
            data_name = "subj" + str(sub).zfill(2) + ".npz"
            o_list = np.load(args.path + data_name)
            total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

        # for sub in list_ :            
        #     data_name = self.make_name("Split", args.test_size, args.class_num, args.expt, 1, str(sub), "_val.npz")
        #     data_name = self.make_name("Single", None, args.class_num, args.expt, 2, str(sub), ".npz")
        #     o_list = np.load(args.path + data_name)
        #     total_list_x.append(o_list['x']); total_list_y.append(o_list['y'])

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