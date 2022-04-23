# import numpy as np
# data = np.load(f"/opt/workspace/xohyun/MS_codes/features_DeepConvNet/original_subj01.npz")
# x_raw = data['arr_0']; y_raw = data['arr_1']
# print(x_raw.shape)
# print(y_raw)
import torch
import numpy as np
import torch.nn as nn
# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(23, 10)
# a batch of 2 samples of 4 indices each
k = torch.zeros((256)).long()
print(k)
print("---",k.shape)
input = torch.LongTensor(k)
embedding(input)


raise
# example with padding_idx
embedding = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0,2,0,5]])
embedding(input)

# example of changing `pad` vector
padding_idx = 0
embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
embedding.weight
with torch.no_grad():
    embedding.weight[padding_idx] = torch.ones(3)
embedding.weight