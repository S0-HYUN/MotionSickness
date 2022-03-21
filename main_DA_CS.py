from sys import stderr
from torch.optim import optimizer
import torch
import pandas as pd
from get_args_CS import Args
from utils import *
import data_loader.data_loader_CS
import data_loader.data_loader_stft
from Model.model_maker import ModelMaker
from trainer.trainer_DA_CS import TrainMaker
# from trainer.trainer_dependent_spectrogram import TrainMaker_spectrogram
from trainer.trainer_dependent_shallow import TrainMaker_shallow
import wandb
from torch.utils.data import Dataset, ConcatDataset

def main():
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
        data = data_loader.data_loader_CS.Dataset(args, phase="train")
        data_valid = data_loader.data_loader_CS.Dataset(args, phase="valid")
        # data_valid = data_loader.data_loader_active.Dataset(args, phase="test")

    # Build model
    model = ModelMaker(args_class).model
    
    # Make trainer
    trainer = TrainMaker(args, model, data, data_valid) # trainer = TrainMaker(args, model, data, None)

    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])

    if args.mode == "train":
        # wandb.init(project=f"{args.model}_{args.standard}_{args.class_num}_case4", 
        #             entity="sohyun", 
        #             name=f"CS_{args.test_subj}_{args.model}_{args.standard}_{args.class_num}")
        wandb.init(project="test", entity="sohyun", name=f"CS_{args.test_subj}_{args.model}_{args.lr}_{args.wd}_{args.scheduler}_{args.optimizer}_{args.class_num}clsss")
        wandb.watch(model, log='all')
        f1_v, acc_v, cm_v, loss_v = trainer.training() # fitting
        
        # current_time = get_time()
        # df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v]
        # df.to_csv(f'./csvs/{current_time}_MS_DA_results_{args.model}_subj{args.test_subj}_train.csv', header = True, index = False)
        
        # print(f"f1:{f1_v}, acc:{acc_v}, loss:{loss_v} \n {cm_v}")
        # if args.model == "ShallowConvNet":
        #     df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v]
        # else:
        #     df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
        
    elif args.DA == False and args.mode == "test":
        data_test = data_loader.data_loader_CS.Dataset(args, phase="test")
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)
        current_time = get_time()
        df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
        df.to_csv(f'./csvs_CS/{current_time}_CS_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class_test.csv', header = True, index = False)

    if args.DA == True:    
        if args.mode == "train":
            model = ModelMaker(args_class, first=False).model
            data_da = data_loader.data_loader_active.Dataset(args, phase="DA")
            trainer = TrainMaker(args, model, data_da)
            f1_v, acc_v, cm_v, loss_v = trainer.training()
            current_time = get_time()
            df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v]
            df.to_csv(f'./csvs_CS/{current_time}_CS_DA_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class.csv', header = True, index = False)

        else:
            data_test_da = data_loader.data_loader_active.Dataset(args, phase="DA_test")
            f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test_da)
            current_time = get_time()
            df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
            df.to_csv(f'./csvs_CS/{current_time}_CS_DA_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class_test.csv', header = True, index = False)

if __name__ == "__main__" :
    main()