from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader_bcic
import pandas as pd
from get_args import Args
from utils import *
from Model.model_maker import ModelMaker
from trainer_dependent import TrainMaker
import wandb

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
    data = data_loader_bcic.Dataset(args, phase="train")
    data_valid = data_loader_bcic.Dataset(args, phase="valid")
    data_test = data_loader_bcic.Dataset(args, phase="test")
    
    # Build model
    model = ModelMaker(args_class).model
    
    # Make trainer
    trainer = TrainMaker(args, model, data, data_valid)
    
    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])

    if args.mode == "train":
        f1_v, acc_v, cm_v, loss_v = trainer.training() # fitting
        print(f"f1:{f1_v}, acc:{acc_v}, loss:{loss_v} \n {cm_v}")
    elif args.mode ==  "test":
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)

    df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
    df.to_csv(f'./csvs/2201141505_bcic_dependent_results_{args.model}_subj{args.test_subj}.csv', header = True, index = False)

if __name__ == "__main__" :
    main()