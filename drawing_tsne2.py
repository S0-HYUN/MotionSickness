import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
# from tsnecuda import TSNE
from utils_drawing import *
from utils import *

x_raw = []
y_raw = []

total_list_x = []; total_list_y = []
for i in range(1,24):
    for j in range(1,3):
        data = np.load(f"/opt/workspace/xohyun/MS_codes/Files_scale/Single/Class3/Expt1/day{j}/subj{i:02d}.npz")
        total_list_x.append(data['x']); total_list_y.append(data['y'])

data_x = np.vstack(total_list_x); data_y = np.vstack(total_list_y)
# data_x = data_x.reshape(-1, 28); data_y = data_y.reshape(-1, 1)
data_x = np.sum(data_x, axis=1)
data_y = np.mean(data_y, axis=1)

visualizer(data_x, data_y)

# for i in range(1,24): #1,24
#     print(f"loading...{i}")
#     data1 = np.load(f"/opt/workspace/xohyun/MS/Files/Single/Class3/Expt1/day1/subj{i:02d}.npz")
#     data2 = np.load(f"/opt/workspace/xohyun/MS/Files/Single/Class3/Expt1/day1/subj{i:02d}.npz")
    
#     x_subset = data1['x']
#     # x_subset = x_subset.reshape(-1, 28)
#     x_subset2 = data2['x']
#     # y_subset = data1['y']

#     # x_subset2 = x_subset2.reshape(-1, 28)
#     x_subset = np.concatenate((x_subset, x_subset2), axis=0)
#     y_subset = np.full(shape=(x_subset.shape[0], 1), fill_value=i)

#     x_subset = torch.tensor(x_subset)
#     x_subset = torch.sum(x_subset, axis=1)
#     x_subset = x_subset.cpu().numpy()

#     if len(x_raw) == 0:
#         x_raw = x_subset
#         y_raw = y_subset
#     else:
#         x_raw = np.concatenate((x_raw, x_subset), axis=0)
#         y_raw = np.concatenate((y_raw, y_subset), axis=0)

# n_components = 2
# transformed_data = TSNE(n_components=n_components).fit_transform(x_raw)
# x_component, y_component = transformed_data[:,0], transformed_data[:,1]

# plt.figure(figsize=(8.5,6), dpi=130)
# # plt.scatter(x=x_component, y=y_component, c=y_raw, cmap="viridis", s=50, alpha=8/10)
# plt.scatter(x=x_component, y=y_component, c=y_raw, cmap=plt.cm.get_cmap("rainbow",23), s=1, alpha=8/10)
# plt.savefig("./tsne1-23_rainbow.png")

