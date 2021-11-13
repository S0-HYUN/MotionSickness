import numpy as np
# tt = np.load("./subj01_day1_train.npz")
# print(tt['y'])
# print(tt['y'].shape)
# print(tt['x'].shape)

tt2 = np.load("/opt/workspace/xohyun/MS/ALoutput/Split0.8/Class3/Expt2/day1/subj01_val.npz")

print("========================", tt2['x'].shape)
print(tt2['x'])
print(tt2['y'].shape)
print(tt2['y'])