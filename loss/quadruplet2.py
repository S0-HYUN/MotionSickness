import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class sosoLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, true_label, embedding, proto, margin1 = 0.5, margin2 = 1):
        # proto = torch.repeat(proto, x1.shape[0]).reshape(-1,3)
        class0_idx = true_label==0; class1_idx = true_label==1; class2_idx = true_label==2
        x1 = embedding[class0_idx]; x2 = embedding[class1_idx]; x3 = embedding[class2_idx]
        count = min(sum(class0_idx), sum(class1_idx), sum(class2_idx))
        total_loss = 0
        if count ==  0 :
            return 0

        for cc in range(count):
            # loss_sep = self.criterion_quad(x1[cc], x2[cc], x3[cc], proto)
        
            # d1 = torch.norm(torch.stack([x1[cc], proto]), p=2) # distance between x1 and proto
            # d2 = torch.norm(torch.stack([x2[cc], proto]), p=2) # distance between x2 and proto
            # d3 = torch.norm(torch.stack([x3[cc], proto]), p=2) # distance between x3 and proto
            d1 = (proto - x1[cc]).pow(2).sum(0)  
            d2 = (proto - x2[cc]).pow(2).sum(0)
            d3 = (proto - x3[cc]).pow(2).sum(0)

            loss1 = F.relu(- d2 + d1 + margin1)
            loss2 = F.relu(- d3 + d2 + margin1)
            loss3 = F.relu(- d3 + d1 + margin2)
            losses = loss1 + loss2 + loss3
            total_loss += losses
            # print("loss print: ",loss1, loss2, loss3)

        return (total_loss / count)