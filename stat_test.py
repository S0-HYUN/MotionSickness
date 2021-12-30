from scipy.stats.stats import mannwhitneyu
from statsmodels.stats.diagnostic import kstest_normal
import numpy as np
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
x_df = pd.DataFrame(data_x)
y_df = pd.DataFrame(data_y, columns=['label'])

choose_idx1 = y_df['label'].isin([1.0])
x_df1 = x_df.loc[choose_idx1]
y_df1 = y_df.loc[choose_idx1]

choose_idx6 = y_df['label'].isin([2.0,3.0,4.0,5.0,6.0,7.0,8.0])
x_df6 = x_df.loc[choose_idx6]
y_df6 = y_df.loc[choose_idx6]

data_x1 = x_df1.to_numpy(); data_y1 = y_df1.to_numpy()
data_x6 = x_df6.to_numpy(); data_y6 = y_df6.to_numpy()

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
principalComponents1 = pca.fit_transform(data_x1)
principalComponents6 = pca.fit_transform(data_x6)
# print(principalComponents1, "\n\n", principalComponents6)

#---# 정규분포성 검정 #---# # test whether a sample differs from a normal distribution
# kstest_normal(data_x, dist='norm')
# from scipy import stats
# k2, p = stats.normaltest(data_x, axis = 0) # null hypothesis can be rejected
# print(k2, "\n", p)

#---# 비모수 #---#
# U1, p = mannwhitneyu(data_x1, data_x6)
U1, p = mannwhitneyu(principalComponents1, principalComponents6)
print(U1, "\n", p)