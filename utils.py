import os
import shutil
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def createFolder(directory) :
    ### create folder ###
    try : 
        if not os.path.exists(directory) :
            os.makedirs(directory)
    except OSError :
        print("Error: Creating directory" + directory)

def prepare_folder(param_path, runs_path) :
    ### remove and create folder if aleady exist###
    if os.path.isdir(param_path) :
        print("[Remove folder]")
        shutil.rmtree(param_path)
    if os.path.isdir(runs_path) :
        shutil.rmtree(runs_path)
    createFolder(param_path)

def gpu_checking() :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    return device

def data_preprocesesing(train, remove, test_list) :
    train_list = [x for x in train if x not in (test_list)]
    train_list = [x for x in train_list if x not in (remove)]
    return train_list

def fix_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Multi-GPU
    if args.device == "multi":
        torch.cuda.manual_seed_all(args.seed)
    # Single-GPU
    else:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False  # If you want to set randomness, cudnn.benchmark = False
    cudnn.deterministic = True  # If you want to set randomness, cudnn.benchmark = True
    print(f"[Control randomness]\nseed: {args.seed}\n")