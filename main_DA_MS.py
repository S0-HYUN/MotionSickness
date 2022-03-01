from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import pandas as pd
from get_args import Args
from utils import *
import data_loader.data_loader_active
import data_loader.data_loader_stft
from Model.model_maker import ModelMaker
from trainer.trainer_DA import TrainMaker
# from trainer.trainer_dependent_spectrogram import TrainMaker_spectrogram
from trainer.trainer_dependent_shallow import TrainMaker_shallow
import wandb
from torch.utils.data import Dataset, ConcatDataset

def main():
    # wandb.init(project="my-test-project", entity="sohyun")
    args_class = Args()
    args = args_class.args

    # Fix seed
    if args.seed:
        fix_random_seed(args)
    
    # Save a file 
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'epoch', 'acc', 'f1', 'loss']); idx = 0

    # Load data
    if args.model == "CRL":
        data = data_loader.data_loader_stft.Dataset(args, phase="train")
        data_valid = data_loader.data_loader_stft.Dataset(args, phase="valid")
        # data_test = data_loader.data_loader_stft.Dataset(args, phase="test")
    else :
        data = data_loader.data_loader_active.Dataset(args, phase="train")
        data_valid = data_loader.data_loader_active.Dataset(args, phase="valid")
        # data_valid = data_loader.data_loader_active.Dataset(args, phase="test")

    # Build model
    model = ModelMaker(args_class).model
    
    # Make trainer
    if args.model == "ShallowConvNet":
        train_set1 = BcicDataset_window(data, window_num=0)
        train_set2 = BcicDataset_window(data, window_num=1) # train_set1.dataset.x.shape
        train_set = ConcatDataset([train_set1, train_set2])
        args.one_bundle = 750 #1000
        trainer = TrainMaker_shallow(args, model, train_set, data_valid)
    else:
        trainer = TrainMaker(args, model, data, data_valid)

    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])

    if args.mode == "train":
        f1_v, acc_v, cm_v, loss_v = trainer.training() # fitting
        # print(f"f1:{f1_v}, acc:{acc_v}, loss:{loss_v} \n {cm_v}")
        # if args.model == "ShallowConvNet":
        #     df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v]
        # else:
        #     df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
        
    elif args.DA == False and args.mode == "test":
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)
        
    if args.DA == True:    
        if args.mode == "train":
            model = ModelMaker(args_class, first=False).model
            data_da = data_loader.data_loader_active.Dataset(args, phase="DA")
            trainer = TrainMaker(args, model, data_da)
            f1_v, acc_v, cm_v, loss_v = trainer.training()
        else:
            data_test_da = data_loader.data_loader_active.Dataset(args, phase="DA_test")
            f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test_da)

    current_time = get_time()
    df.to_csv(f'./csvs/{current_time}_MS_independent_results_{args.model}_subj{args.test_subj}.csv', header = True, index = False)

class BcicDataset_window(Dataset):
    def __init__(self, dataset, window_num):
        self.dataset = dataset
        self.len = len(dataset)
        self.window_num = window_num

    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        X, y = self.dataset.__getitem__(idx)
        return X[:,self.window_num*125:self.window_num*125+1000], y
        # X, y, idx = self.dataset.__getitem__(idx)
        # return X[:,:,self.window_num*125:self.window_num*125+1000], y, idx

if __name__ == "__main__" :
    main()