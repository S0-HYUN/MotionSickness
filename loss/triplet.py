import torch.nn.functional as F
import torch.nn as nn
import torch

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, proto, embedding, true_label, margin=1):
        class_pos_idx = true_label==0; class_neg_idx = true_label != 0
        pos = embedding[class_pos_idx]; neg = embedding[class_neg_idx]
        count = min(sum(class_pos_idx), sum(class_neg_idx))
        total_loss = 0
    
        for cc in range(count):
            # loss_sep = self.criterion_quad(x1[cc], x2[cc], x3[cc], proto)
            distance_positive = (proto - pos[cc]).pow(2).sum(0)  # .pow(.5)
            distance_negative = (proto - neg[cc]).pow(2).sum(0)  # .pow(.5)
            losses = F.relu(distance_positive - distance_negative + margin)
            total_loss += losses
        return (total_loss / count)
        # return losses.mean() if size_average else losses.sum()
    
    # def __init__(self, margin):
    #     super(TripletLoss, self).__init__()
    #     self.margin = margin

    # def forward(self, anchor, positive, negative, size_average=True):
    #     distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    #     distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    #     losses = F.relu(distance_positive - distance_negative + self.margin)
    #     return losses.mean() if size_average else losses.sum()