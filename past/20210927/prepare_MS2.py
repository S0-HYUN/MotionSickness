import numpy as np
tt = np.load("./subj01_day1_train.npz")
print(tt['y'])
print(tt['y'].shape)
print(tt['x'].shape)

tt2 = np.load("./subj01_day1_val.npz")
print(tt2['y'])
print(tt2['y'].shape)
print(tt2['x'].shape)
