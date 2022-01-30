from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader.data_loader_stft
import pandas as pd
from get_args import Args
from utils import *
from Model.model_maker import ModelMaker
from trainer_dependent import TrainMaker
from trainer_dependent_shallow import TrainMaker_shallow
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
    data = data_loader.data_loader_stft.Dataset(args, phase="train")
    data_valid = data_loader.data_loader_stft.Dataset(args, phase="valid")
    # data_test = data_loader.data_loader_stft.Dataset(args, phase="test")
    
    # Build model
    model = ModelMaker(args_class).model
    
    # Make trainer
    if args.model == "ShallowConvNet":
        train_set1 = BcicDataset_window(data, window_num=0)
        train_set2 = BcicDataset_window(data, window_num=1)
        train_set = ConcatDataset([train_set1, train_set2])
        args.one_bundle = 1000
        trainer = TrainMaker_shallow(args, model, train_set, data_valid)
    else:
        trainer = TrainMaker(args, model, data, data_valid)

    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])

    if args.mode == "train":
        f1_v, acc_v, cm_v, loss_v = trainer.training() # fitting
        print(f"f1:{f1_v}, acc:{acc_v}, loss:{loss_v} \n {cm_v}")
        if args.model == "ShallowConvNet":
            df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v]
        else:
            df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
    elif args.mode ==  "test":
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)

    df.to_csv(f'./csvs/2201181200_bcic_dependent_results_{args.model}_subj{args.test_subj}.csv', header = True, index = False)

if __name__ == "__main__" :
    main()