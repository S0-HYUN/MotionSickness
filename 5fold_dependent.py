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
    wandb.init(project="my-test-project", entity="sohyun")
    args_class = Args()
    args = args_class.args

    # Save a file
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'acc', 'f1', 'loss']); idx = 0

    sub = args.test_subj; print(f"==={args.test_subj}===")

    total_list_x = []
    for d in range(1,3): 
        data_name = make_name("Single", None, args.class_num, args.expt, d, str(sub), ".npz")
        o_list = np.load(args.path + data_name)
        total_list_x.append(o_list['x'])

    data_x = np.vstack(total_list_x)
    num = data_x.shape[0]; print("총개수", num)

    from sklearn.model_selection import KFold
    num_range = np.arange(num); random.shuffle(num_range) # shuffle
    ss = KFold(n_splits=5)
    for train_index, test_index in ss.split(num_range):
        # data = data_loader.Dataset(args, phase="train", idx=train_index[:int(len(train_index)*0.75)])
        # data_valid = data_loader.Dataset(args, phase="valid", idx=train_index[int(len(train_index)*0.75):])
        # data_test = data_loader.Dataset(args, phase="test", idx=test_index)
        data = data_loader.Dataset(args, phase="train", idx=train_index)
        data_valid = data_loader.Dataset(args, phase="train", idx=test_index)

        model = ModelMaker(args_class).model
        trainer = TrainMaker(args, model, data, data_valid)
        f1_v, acc_v, cm_v, loss_v = trainer.training()

        # args.mode = "test"
        # model_test = ModelMaker(args_class).model
        # trainer_test = TrainMaker(args, model_test, data_test)
        # f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)

        df.loc[idx] = [args.test_subj, args.lr, args.wd, acc_v, f1_v, loss_v.cpu().numpy()]
        df.to_csv(f'./csvs/Dependent_5fold_results_{args.model}_subj{args.test_subj}_batchsampler_parameter.csv', header = True, index = False)
        idx += 1

def make_name(category_ss, test_size, class_num, expt, day, subj_num, category_tv):
        '''
        make data file name function
        input     : category_ss [Split / Single], test_size, num of class, expt, day, subj_num, category_tv [train / valid]
        output    : data file name
        '''
        if test_size :
            data_name = category_ss + str(test_size)
        else :
            data_name = category_ss
        
        data_name = data_name + \
                    "/Class" + str(class_num) + \
                    "/Expt" + str(expt) + \
                    "/day" + str(day) + \
                    "/subj" + subj_num.zfill(2) + category_tv
        return data_name

if __name__ == "__main__" :
    main()