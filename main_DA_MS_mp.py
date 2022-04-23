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
# from trainer.trainer_DA import TrainMaker
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from trainer.trainer_DA import TrainMaker as TrainMaker_soso
from trainer.trainer_DA_baseline import TrainMaker as TrainMaker_baseline
from trainer.trainer_DA_baseline_MSE import TrainMaker as TrainMaker_baseline_MSE
from trainer.trainer_DA_soso_MSE import TrainMaker as TrainMaker_soso_MSE
from trainer.trainer_DA_soso_quadruplet import TrainMaker as TrainMaker_soso_quadruplet
from trainer.trainer_DA_soso_quadruplet_direct4 import TrainMaker as TrainMaker_soso_quadruplet_direct
from trainer.trainer_DA_soso_triplet_direct import TrainMaker as TrainMaker_soso_triplet_direct
# from trainer.trainer_dependent_spectrogram import TrainMaker_spectrogram
from trainer.trainer_dependent_shallow import TrainMaker_shallow
import wandb
from torch.utils.data import Dataset, ConcatDataset
import torch.nn as nn

def main():
    args_class = Args()
    args = args_class.args

    # Fix seed
    if args.seed:
        fix_random_seed(args)

    # Save a file 
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'epoch', 'acc', 'f1', 'loss']); idx = 0

    # Load data
    if args.model == "CRL": # using spectrogram
        data = data_loader.data_loader_stft.Dataset(args, phase="train")
        data_valid = data_loader.data_loader_stft.Dataset(args, phase="valid")
        # data_test = data_loader.data_loader_stft.Dataset(args, phase="test")
    else :
        data = data_loader.data_loader_active.Dataset(args, phase="train")
        data_rest = data_loader.data_loader_active.Dataset(args, phase="train", rest=True)
        data_valid = data_loader.data_loader_active.Dataset(args, phase="valid")
        data_valid_rest = data_loader.data_loader_active.Dataset(args, phase="valid", rest=True)
        # data_valid = data_loader.data_loader_active.Dataset(args, phase="test")

    # Build model
    model = ModelMaker(args_class).model

    import tracemalloc
    tracemalloc.start()
    
    # Make trainer
    if args.model in ['DeepConvNet', 'ShallowConvNet', 'EEGNet'] :
        if args.criterion == "MSE":
            trainer = TrainMaker_baseline_MSE(args, model, data, data_valid)
        else :
            trainer = TrainMaker_baseline(args, model, data, data_valid) # trainer = TrainMaker(args, model, data, None)
    
    elif args.model == "soso":
        if args.criterion == "MSE":
            trainer = TrainMaker_soso_MSE(args, model, data, data_valid, data_rest, data_valid_rest)
        elif args.criterion == "quad":
            trainer = TrainMaker_soso_quadruplet_direct(args, model, data, data_valid, data_rest, data_valid_rest)
        elif args.criterion == "triplet":
            trainer = TrainMaker_soso_triplet_direct(args, model, data, data_valid, data_rest, data_valid_rest)    
        else :
            trainer = TrainMaker_soso(args, model, data, data_valid)

    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])

    if args.mode == "train":
        wandb.init(project=f"{args.model}_{args.standard}_{args.class_num}_case6", entity="sohyun", name=f"{args.test_subj}_{args.model}_{args.standard}_{args.class_num}_{args.scheduler}")
        # wandb.init(project="test", entity="sohyun", name=f"{args.test_subj}_{args.model}_{args.lr}_{args.wd}_{args.scheduler}_{args.optimizer}_{args.class_num}clsss")
        f1_v, acc_v, cm_v, loss_v = trainer.training() # fitting
        print("-")
    elif args.DA == False and args.mode == "test":
        data_test = data_loader.data_loader_active.Dataset(args, phase="test")
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)
        drawing_cm(args, cm_v)
        current_time = get_time()
        df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
        create_folder(f"./csvs/csvs_{args.model}_{args.standard}_{args.class_num}") # make folder
        df.to_csv(f'./csvs/csvs_{args.model}_{args.standard}_{args.class_num}/{current_time}_MS_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class_test.csv', header = True, index = False)
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[Top 10 ]")
    for stat in top_stats[:10]:
        print(stat) 

if __name__ == "__main__" :
    main()



'''
if args.DA == True:    
        if args.mode == "train":
            model = ModelMaker(args_class, first=False).model
            data_da = data_loader.data_loader_active.Dataset(args, phase="DA")
            trainer = TrainMaker(args, model, data_da)
            f1_v, acc_v, cm_v, loss_v = trainer.training()
            current_time = get_time()
            df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v]
            df.to_csv(f'./csvs/{current_time}_MS_DA_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class.csv', header = True, index = False)

        else:
            data_test_da = data_loader.data_loader_active.Dataset(args, phase="DA_test")
            f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test_da)
            current_time = get_time()
            df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
            df.to_csv(f'./csvs/{current_time}_MS_DA_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class_test.csv', header = True, index = False)
'''