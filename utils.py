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

def data_preprocesesing(train, test_list, remove=None) :
    train_list = [x for x in train if x not in (test_list)]
    if remove != None:
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
        str(current_time.hour + 9).zfill(2) + \
        str(current_time.minute).zfill(2)
    return current

class FeatureExtractor() :
    features =  None
    def __init__(self, m) :
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu()
        self.features_before = input[0].detach().cpu()

def get_meanPrototype(feature=False) :
    if feature :
        idx = 1
        rest_feature = np.load(f"/opt/workspace/xohyun/MS_codes/features_rest(embedding)/original_subj{str(idx).zfill(2)}.npz")
        X_raw = rest_feature['arr_0']; Y_raw = rest_feature['arr_1']; x_value = X_raw; y_value = Y_raw
        for idx in range(2, 24):
            rest_feature = np.load(f"/opt/workspace/xohyun/MS_codes/features_rest(embedding)/original_subj{str(idx).zfill(2)}.npz")
            X_raw = rest_feature['arr_0']; Y_raw = rest_feature['arr_1']
            x_value = np.concatenate((x_value, X_raw)); y_value = np.concatenate((y_value, Y_raw)) # 전체 값 모으기
        mean_proto = x_value.mean(0)
    else :
        import json
        with open('rest_prototype.json') as f:
            prototypes = json.load(f) # 각 subject의 prototypes load

        prototype = list(prototypes.values())
        mean_proto = prototype[0] 
        for i in prototype: mean_proto = np.concatenate((mean_proto, i), axis=0)
        mean_proto = mean_proto.mean(axis=0) # 완전한 mean
    return mean_proto

def change_vector(y) :
    modified_target = [[0 for col in range(3)] for row in range(y.shape[0])]
    for i, target in enumerate(y) :
        for j in range(int(target.item()+1)) :
            modified_target[i][j] = 1
    return torch.tensor(modified_target)

def drawing_cm(args, cm):
    import matplotlib.pyplot as plt
    # title = "dsf"
    print(len(cm))
    cmap=plt.cm.Greens
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # , cmap=plt.cm.Greens
    # plt.title(title, size=12)
    plt.colorbar(fraction=0.05, pad=0.05)
    if len(cm) == 3:
        tick_marks = np.arange(3, 3)
        plt.xticks(np.arange(3), ('0', '1', '2'))
        plt.yticks(np.arange(3), ('0', '1', '2'))
    else:
        tick_marks = np.arange(3, 3)
        plt.xticks(np.arange(2), ('0', '1'))
        plt.yticks(np.arange(2), ('0', '1'))

    fmt = 'd' 
    thresh = 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", color="white" if cm[i, j] > thresh else "black")  #horizontalalignment
    create_folder(f"./cm/cm_{args.model}__{args.criterion}")
    plt.savefig(f"./cm/cm_{args.model}__{args.criterion}/cm_{args.test_subj}")
    

def memory() :
    import tracemalloc

    tracemalloc.start()

    # ... run your application ...

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')

    print("[Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    raise
