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
import pandas as pd

x_raw = []
y_raw = []
expt = 1

total_list_x = []; total_list_y = []
for i in range(1,24):
    for j in range(1,3):
        data = np.load(f"/opt/workspace/xohyun/MS_codes/Files_scale_raw_original/Single/Class3/Expt{expt}/day{j}/subj{i:02d}.npz")
        total_list_x.append(data['x']); total_list_y.append(data['y'])

data_x = np.vstack(total_list_x); data_y = np.vstack(total_list_y)
# data_x = data_x.reshape(-1, 28); data_y = data_y.reshape(-1, 1)
data_x = np.sum(data_x, axis=1)
data_y = np.mean(data_y, axis=1)

#---# choose label #---#
# x_df = pd.DataFrame(data_x)
# y_df = pd.DataFrame(data_y, columns=['label'])

# choose_idx = y_df['label'].isin([5.0,6.0])
# y_df = y_df.loc[choose_idx]
# x_df = x_df.loc[choose_idx]
# # print(x_df)
# # print("\n", y_df)
# data_x = x_df.to_numpy()
# data_y = y_df.to_numpy()

visualizer(data_x, data_y)