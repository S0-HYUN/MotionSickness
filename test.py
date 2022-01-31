# total_budget = 300
# num_rounds = 30
# sampling_ratio = [(total_budget/num_rounds) * n for n in range(num_rounds+1)]
# print(sampling_ratio)

'''
'''

import torch.nn as nn
import numpy as np

# define the helper function
def findConv2dOutShape(H_in, W_in, conv, pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    H_out = np.floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1) / stride[0] + 1)
    W_out = np.floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1) / stride[1] + 1)

    if pool:
        H_out /= pool
        W_out /= pool
    
    return int(H_out), int(W_out)

conv1 = nn.Conv2d(8, 16, kernel_size=(1,28))
h,w = findConv2dOutShape(64,53,conv1)
print(h,w)