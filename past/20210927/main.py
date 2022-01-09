from torch.optim import optimizer
import torch
import data_loader_5fold
import trainer
import os
import pandas as pd
from setting import *
import Network.EEGNet_models
import Network.DeepConvNet_models
import filing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

test_file = filing.calc_test_file(test_subj, expt)
#val_list = filing.calc_test_file(val_subj, expt)
# train_list.remove(test_file)

df = pd.DataFrame(columns = ['test_subj', 'expt', 'lr', 'wd', 'acc', 'f1', 'loss'])
idx = 0

for _ in range(92) :
    train_list = [_]
    val_list = [_]
    data = data_loader_5fold.Dataset(path, train_list, val_list, data_type = 'train') # 73 / 50
    data_validation = data_loader_5fold.Dataset(path, train_list, val_list, data_type = 'valid') # valid 48(13번 day1 expt1) 10개 -> 50(13번 day2 expt1)

    model = Network.EEGNet_models.EEGNet(class_num, channel_num, one_bundle).to(device = device)
    # model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device = device)
    # model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)    

    for lr in lr_list :
        for wd in wd_list :
            print("\n====================================lr{}, wd{}====================================".format(lr, wd))

            trainer1 = trainer.trainer(model, epoch, lr, optimizer = 'Adam', weight_decay=wd)
            acc, f1, cm, loss = trainer1.only_train(data, batch_size=batch_size)

            last_param = os.listdir('./param/lr{}_wd{}/'.format(lr, wd))[-1] ############################################# 저장 위치
            best_model = Network.EEGNet_models.EEGNet(class_num, channel_num, one_bundle).to(device=device)
            # best_model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device=device)
            # best_model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)
            best_model.load_state_dict(torch.load('./param/lr{}_wd{}/'.format(lr, wd)+last_param)) 

            trainer2 = trainer.trainer(best_model, 1, 0.1, 'Adam', 0.1)
            acc_v, f1_v, cm_v, loss_v = trainer2.prediction(data_validation, batch_size, test=True)

            df.loc[idx] = [train_list[0], expt, lr, wd, acc_v, f1_v, loss_v.cpu().numpy()]

            idx += 1 
            df.to_excel('results.xlsx', header = True, index = False)