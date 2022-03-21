from numpy.core.fromnumeric import reshape, transpose
import torch
import torch.nn as nn
import numpy as np
from utils import gpu_checking

class DeepConvNet_dk(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet_dk, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1), # 10 -> 5 # 28 * 741
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
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                # nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        # out = torch.zeros(1, 1, input_ch, input_time)
        #
        # for i, module in enumerate(self.convnet):
        #     print(module)
        #     out = module(out)
        #     print(out.size())
        #

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]

        ##############기억
        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes))
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # # output = self.l2normalize(output)
        output=self.clf(output) 
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class soso(nn.Module):
    def __init__(self, args):
        super(soso, self).__init__()
        self.device = gpu_checking(args)
        self.model = DeepConvNet_dk(args.class_num, args.channel_num, args.one_bundle).to(device = self.device) 
        
        self.dist = args.distance

        import json
        with open('rest_prototype.json') as f:
            prototypes = json.load(f)
      
        self.prototype = list(prototypes.values())
        mean_proto = self.prototype[0]
        for i in self.prototype: mean_proto = np.concatenate((mean_proto, i), axis=0)

        self.mean_proto = mean_proto.mean(axis=0) # 완전한 mean

        # flag = list(self.model._modules)[-1]
        # final_layer = self.model._modules.get(flag)
        # activated_features = FeatureExtractor(final_layer)
    def forward(self, x):
        embeddings = self.model(x.to(device=self.device).float()) # (256, 200, 1, 4) / (batch size, class_num)

        # print(embeddings); print(embeddings.shape)
        
        # if self.dist == "cosine":
        #     dists = 1 - nn.CosineSimilarity(dim=-1)()
        # else:
        #     dists = torch.norm()
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        input1 = embeddings.mean(axis=0).to(dtype=torch.float64).float().unsqueeze(dim=0)
        input2 = torch.tensor(self.mean_proto).to(self.device).float().unsqueeze(dim=0)

        output = pdist(input1, input2)
        print("===거리", output)
        return output