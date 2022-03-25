import torch.nn as nn
import torch

embedding = nn.Embedding(10,3)
input = torch.LongTensor([[1,3,4,5],[4,3,2,9]])

print(input)
print(embedding(input).shape)
# print(embedding.weight)