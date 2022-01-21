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

data = np.load(f"/opt/workspace/xohyun/MS_codes/features_EEGNet_10/original_subj01_epoch10.npz")
x_raw = data['arr_0']; y_raw = data['arr_1']

for i in range(2,24):
    data = np.load(f"/opt/workspace/xohyun/MS_codes/features_EEGNet_10/original_subj{str(i).zfill(2)}_epoch10.npz")
    # x_raw = torch.cat((x_raw, data['arr_0']))
    x_raw = np.concatenate((x_raw, data['arr_0']))
    y_raw = np.concatenate((y_raw, data['arr_1']))

print(y_raw)
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

visualizer(x_raw, y_raw)