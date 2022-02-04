import os
import shutil
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def create_folder(directory) :
    #---# create folder #---#
    try : 
        if not os.path.exists(directory) :
            os.makedirs(directory)
    except OSError :
        print("Error: Creating directory" + directory)

def prepare_folder(path_list) :
    #---# remove and create folder if aleady exist #---#
    for pl in path_list :
        # if os.path.isdir(pl) :
        #     print(f"[Remove folder] {pl}")
        #     shutil.rmtree(pl)
        create_folder(pl)

def gpu_checking(args) :
    device = torch.device(f'cuda:{str(args.device)}' if torch.cuda.is_available() else 'cpu')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
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
    print(f"[Control randomness]\nseed: {args.seed}")

from datetime import datetime
def get_time() :
    current_time = datetime.now()
    year = current_time.year % 100
    current = str(year) + \
        str(current_time.month).zfill(2) + \
        str(current_time.day).zfill(2) + \
        str(current_time.hour).zfill(2) + \
        str(current_time.minute).zfill(2)
    return current

class FeatureExtractor():
    features =  None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu()