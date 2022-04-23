from cmath import inf
import torch
import torch.nn as nn
import numpy as np

class ODML(nn.Module):
    def __init__(self, k=4, v=2):
        super(ODML, self).__init__()
        self.k = k # number of target neighbors
        self.v = v # number of neighbors with different class labels
    def forward(self, x, y):
        label = np.unique(y)
        n = y.shape[0]
        x = x.reshape(-1, x.shape[0])
    
        D = torch.sum(torch.pow(x, 2), 0); D = D.unsqueeze(0)
        D_ = torch.transpose(D, 0, 1) + (-2) * torch.mm(torch.transpose(x, 0, 1), x)
        D = D + D_
        D[list(range(0,x.shape[1],n+1))] = float("inf")
        T = torch.zeros(3, n * (self.v + self.k) * (self.v + self.k))
        m = 0

        idxs = []
        for i in range(label) : # find targets
            for idx in y.shape[0] :
                if y[idx] == i :
                    idxs.append(idx)
                    
        print()   
        ### 3차원으로 도전하려다가 실패함.
        # D = []
        # for idx in range(x.shape[0]) :
        #     D_ = torch.sum(torch.pow(x[idx, :, :], 2))
        #     D.append(D_)
        # D = torch.tensor(D); D = D.unsqueeze(0)
        # s_ = D.transpose(0, 1) + (-2) * torch.mm(x.transpose(0,1), x) + D
        # print(D[0])


if __name__ == '__main__':
    x = torch.rand(4, 750, 28)
    y = torch.tensor([2,1,2,0])
    
    model = ODML()
    m = model(x, y)