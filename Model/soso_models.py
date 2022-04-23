from numpy.core.fromnumeric import reshape, transpose
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from zmq import device
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import gpu_checking

class soso(nn.Module):
    def __init__(self, args, n_subj=None):
        super(soso, self).__init__()
        self.device = gpu_checking(args)
        self.args = args
        ###
        if n_subj : self.n_subjs = n_subj
        ran_channels = 10
        upsample_initial_channel = 800 # shape 1D
        ## deepconvnet
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        n_ch4 = 200
        input_ch = self.args.channel_num
        self.batch_norm=True
        self.batch_norm_alpha=0.1
        self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1), # 10 -> 5 # 28 * 741 # (10, 247)
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)), #28 238

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)), #28 76

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                )

        self.embedding = nn.Embedding(self.n_subjs, ran_channels)
        self.cond = nn.Conv1d(ran_channels, upsample_initial_channel, 1)
        self.fc = nn.Linear(800, self.args.class_num)
        self.fc2 = nn.Sequential(nn.Linear(800, self.args.class_num), nn.Sigmoid()) # cat으로 붙이면 840

    def forward(self, x, subj=None, embedding = False, fc =  False):
        x = self.convnet(x).squeeze(2)
        
        if embedding :
            subj = subj - 1 # id
            g = self.embedding(subj).unsqueeze(-1) # [batch, h, 1] [256,10] #[256, 10, 1]
            if g is not None : 
                emd = self.cond(g)
                emd = emd.reshape(x.shape[0], -1, x.shape[2])
                # x = torch.cat([x, emd], dim=1)
                x = x + emd
                x = x.reshape(x.shape[0], -1)
            if fc : x = self.fc2(x)
        else :
            x = x.reshape(x.shape[0], -1) # [256, 800]
            if fc : x = self.fc(x)
        return x

    # def forward(self, x, subj):
    #     subj = subj - 1 # id
    #     g = self.embedding(subj).unsqueeze(-1) # [batch, h, 1] [256,10] #[256, 10, 1]

    #     if g is not None : 
    #         x = self.convnet(x).squeeze(2) + self.cond(g)
 
    #     x = x.reshape(x.shape[0], -1) # [256, 800]
    #     x = self.fc(x)
    #     return x

        #######
        # index_0 = (y==0); index_1 = (y==1); index_2 = (y==2)
        
        # pdist = nn.PairwiseDistance(p=2)
        # d_c0 = pdist(embeddings[index_0], torch.tensor(self.mean_proto).to(self.device).float().unsqueeze(dim=0))
        # d_c1 = pdist(embeddings[index_1], torch.tensor(self.mean_proto).to(self.device).float().unsqueeze(dim=0))
        # d_c2 = pdist(embeddings[index_2], torch.tensor(self.mean_proto).to(self.device).float().unsqueeze(dim=0))
        # print("===",torch.mean(d_c0)); print("===",torch.mean(d_c1)); print("===",torch.mean(d_c2))
        # dis = torch.zeros(embeddings.shape[0], device=self.device)
        # dis[index_0] = d_c0; dis[index_1] = d_c1; dis[index_2] = d_c2; dis = dis.unsqueeze(dim=0)
        
        # dis_normal = F.normalize(dis) # shape : (1, batch_size)
        
        #######
        # output = self.conv(dis_normal)
        # print(output)
        
        #######
        # if self.dist == "cosine":
        #     dists = 1 - nn.CosineSimilarity(dim=-1)()
        # else:
        #     dists = torch.norm()
        
        ###
        # pdist = nn.PairwiseDistance(p=2, keepdim=True)
        # input1 = embeddings.mean(axis=0).to(dtype=torch.float64).float().unsqueeze(dim=0)
        # input2 = torch.tensor(self.mean_proto).to(self.device).float().unsqueeze(dim=0)

        # output = pdist(input1, input2)

        # return output