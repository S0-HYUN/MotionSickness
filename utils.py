import os
import shutil
import torch

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