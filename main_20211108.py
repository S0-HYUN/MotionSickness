from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader
import trainer
import os
import pandas as pd
import Network.EEGNet_models
import Network.DeepConvNet_models
from get_args import Args
import subprocess
from utils import *

def main():
    args_class = Args()
    args = args_class.args

    device = gpu_checking()
 
    if not os.path.isdir(args.path + "Split" + str(args.test_size)) :
        process = subprocess.Popen(["python", "/opt/workspace/soxo/soh_code_f/prepare_MS.py", "-test_size", str(args.test_size)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.communicate()

    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'acc', 'f1', 'loss']); idx = 0
    
    train_list = data_preprocesesing(list(range(1,24)), args.remove_subj, [args.test_subj])

    data = data_loader.Dataset(args, args.path, train_list, [args.test_subj], data_type = 'train')
    data_valid = data_loader.Dataset(args, args.path, None, [args.test_subj], data_type = 'valid')
    data_test = data_loader.Dataset(args, args.path, train_list, [args.test_subj], data_type = 'test')

    model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device = device)
    # model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device = device)
    # model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)    

    prepare_folder(args.param_path, args.runs_path)

    for lr in args.lr_list :
        for wd in args.wd_list :
            print("\n====================================lr{}, wd{}====================================".format(lr, wd))

            trainer1 = trainer.trainer(args, model, args.epoch, lr, optimizer = 'Adam', weight_decay=wd)
            # acc, f1, cm, loss = trainer1.only_train(data, batch_size=args.batch_size)
            acc, f1, cm, loss = trainer1.train(data, data_valid, args.batch_size)
            last_param = os.listdir('./param/lr{}_wd{}/'.format(lr, wd))[-1] ############################################# 저장 위치
            best_model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device=device)
            # best_model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device=device)
            # best_model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)
            best_model.load_state_dict(torch.load('./param/lr{}_wd{}/'.format(lr, wd)+last_param)) 

            trainer2 = trainer.trainer(args, best_model, 1, 0.1, 'Adam', 0.1)
            acc_v, f1_v, cm_v, loss_v = trainer2.prediction(data_test, args.batch_size, test=True)

            df.loc[idx] = [args.test_subj, lr, wd, acc_v, f1_v, loss_v.cpu().numpy()]
 
            idx += 1 
            df.to_excel('results_EEGNET_subj{}.xlsx'.format(args.test_subj), header = True, index = False)


if __name__ == "__main__" :
    main()

