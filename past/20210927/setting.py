import os
import filing
path = '/opt/workspace/soxo/Motionsickness_Data/PREPROCESSED_DATA/' # path = './Motionsickness Data/PREPROCESSED_DATA/'
data_list = os.listdir(path)


# test_day = 1
test_subj = 1        #13
expt = 1             # 1:오전 2:오후
if expt == 1 :  train_list = [i for i in range(len(data_list)) if i % 2 == 0]
else :          train_list = [i for i in range(len(data_list)) if i % 2 == 1]

batch_size = 16
epoch = 100
lr_list = [1e-5, 1e-4, 1e-3]
wd_list = [1e-5, 1e-4, 1e-3]

# 추 ###############################################################################################
# lr_list = [5e-5,1e-4]
# wd_list = [1e-1, 1e-3]
# train_list = [50]
# val_list = [filing.calc_test_file(13, 1)]
# train_list = val_list = [0]
test_size = 0.5 # validation test set 비율

one_bundle = int(1500/2)
channel_num = 28
class_num = 3
lower_freq = 0.5
high_freq = 50
score_list = [3, 7] 
# [3] -> 0,1,2,3 /4,5,6,7,8,9
# [3,7] -> 0,1,2,3 / 4,5,6 / 7,8,9



# 500hz -> 30초에 15000행
# 3초에 1500행
