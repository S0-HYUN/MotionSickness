from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import pandas as pd
from get_args import Args
from utils import *
import data_loader.data_loader_FTL
from Model.model_maker import ModelMaker
# from trainer.trainer_DA import TrainMaker
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from trainer.trainer_FTL import TrainMaker
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
    data = data_loader.data_loader_FTL.Dataset(args, phase="train")
    data_target = data_loader.data_loader_FTL.Dataset(args, phase="valid")
    data_test = data_loader.data_loader_FTL.Dataset(args, phase="test")

    # # Build model
    # model = ModelMaker(args_class).model
    
    # Make trainer
    trainer = TrainMaker(args, data, data_target, data_test) # trainer = TrainMaker(args, model, data, None)

    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])
 
    if args.mode == "train":
        # wandb.init(project=f"{args.model}_{args.standard}_{args.class_num}_case6", entity="sohyun", name=f"{args.test_subj}_{args.model}_{args.standard}_{args.class_num}_{args.scheduler}")
        # wandb.init(project="test", entity="sohyun", name=f"{args.test_subj}_{args.model}_{args.lr}_{args.wd}_{args.scheduler}_{args.optimizer}_{args.class_num}clsss")
        acc = trainer.training() # fitting
        print("====", acc)
        raise

    elif args.DA == False and args.mode == "test":
        data_test = data_loader.data_loader_active.Dataset(args, phase="test")
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)
        current_time = get_time()
        df.loc[idx] = [args.test_subj, args.lr, args.wd, args.epoch, acc_v, f1_v, loss_v.cpu().numpy()]
        create_folder(f"./csvs/csvs_{args.model}_{args.standard}_{args.class_num}") # make folder
        df.to_csv(f'./csvs/csvs_{args.model}_{args.standard}_{args.class_num}/{current_time}_MS_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class_test.csv', header = True, index = False)

if __name__ == "__main__" :
    main()