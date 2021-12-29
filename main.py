from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader
import trainer
import os
import pandas as pd
from get_args import Args
import subprocess
from utils import *
from Model.model_maker import ModelMaker
from trainer import TrainMaker
import wandb

def main():
    # wandb.init(project="my-test-project", entity="sohyun")
    args_class = Args()
    args = args_class.args
 
    # Fix seed
    if args.seed:
        fix_random_seed(args)
    
    # Save a file 
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'acc', 'f1', 'loss']); idx = 0

    # Load data
    data = data_loader.Dataset(args, phase="train")
    data_valid = data_loader.Dataset(args, phase="valid")
    data_test = data_loader.Dataset(args, phase="test")
    
    # Build model
    model = ModelMaker(args_class).model

    # Make trainer
    trainer = TrainMaker(args, model, data, data_valid)
    
    # Prepare folder
    prepare_folder(args.param_path, args.runs_path)

    if args.mode == "train":
        trainer.training() # fitting
    elif args.mode ==  "test":
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)
    # acc, f1, cm, loss = trainer1.only_train(data, batch_size=args.batch_size)
    # acc, f1, cm, loss = trainer1.train(data, args.batch_size)
    last_param = os.listdir('./param/lr{}_wd{}/'.format(args.lr, args.wd))[-1] ############################################# 저장 위치
    
    '''
    best_model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device=device)
    # best_model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device=device)
    # best_model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)
    best_model.load_state_dict(torch.load('./param/lr{}_wd{}/'.format(args.lr, args.wd)+last_param)) 

    trainer2 = trainer.trainer(args, best_model, 1, 0.1, 'Adam', 0.1)
    acc_v, f1_v, cm_v, loss_v = trainer2.prediction(data_test, args.batch_size, test=True)
    '''

    df.loc[idx] = [args.test_subj, args.lr, args.wd, acc_v, f1_v, loss_v.cpu().numpy()]
    df.to_excel('results_{}_subj{}.xlsx'.format(args.model, args.test_subj), header = True, index = False)

if __name__ == "__main__" :
    main()






# for lr in args.lr_list :
#         for wd in args.wd_list :
#             print("\n====================================lr{}, wd{}====================================".format(lr, wd))

#             trainer1 = trainer.trainer(args, model, lr, weight_decay=wd)
#             # acc, f1, cm, loss = trainer1.only_train(data, batch_size=args.batch_size)
#             acc, f1, cm, loss = trainer1.train(data, args.batch_size)
#             last_param = os.listdir('./param/lr{}_wd{}/'.format(lr, wd))[-1] ############################################# 저장 위치
            
            
#             best_model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device=device)
#             # best_model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device=device)
#             # best_model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)
#             best_model.load_state_dict(torch.load('./param/lr{}_wd{}/'.format(lr, wd)+last_param)) 

#             trainer2 = trainer.trainer(args, best_model, 1, 0.1, 'Adam', 0.1)
#             acc_v, f1_v, cm_v, loss_v = trainer2.prediction(data_test, args.batch_size, test=True)

#             df.loc[idx] = [args.test_subj, lr, wd, acc_v, f1_v, loss_v.cpu().numpy()]
 
#             idx += 1 
#             df.to_excel('results_{}_subj{}.xlsx'.format(args.model, args.test_subj), header = True, index = False)