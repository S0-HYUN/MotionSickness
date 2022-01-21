from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import pandas as pd
from get_args import Args
from utils import *
from new_trainer import *
from torch.utils.data import Dataset, ConcatDataset
import wandb
import pickle
from Model.DeepConvNet_models import ShallowConvNet_dk
import data_loader_bcic
        

def main():
    # wandb.init(project="my-test-project", entity="sohyun")
    args_class = Args()
    args = args_class.args
 
    # Fix seed
    if args.seed:
        fix_random_seed(args)

    # Build model
    # model = ModelMaker(args_class).model

    # Prepare folder
    prepare_folder([args.param_path, args.runs_path])

    n_classes = 4
    subject_id = args.test_subj
    data_root = "./"

    # with open(data_root+ 'bcic_datasets_[0,38].pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print('subject:' + str(subject_id))
    '''
    total_list_x = []; total_list_y = []

    for sub in [args.test_subj]:
        data_name = "subj" + str(sub).zfill(2) + ".npz"
        o_list = np.load(args.path + data_name)
        data_len = o_list['x'].shape[0]
        temp_x = o_list['x'][:int(data_len/2)]
        temp_y = o_list['y'][:int(data_len/2)]
        total_list_x.append(temp_x); total_list_y.append(temp_y)

    data_x, data_y = np.vstack(total_list_x), np.vstack(total_list_y)
    x = torch.tensor(data_x)
    y = torch.tensor(data_y)
    y = y.reshape(y.shape[1])

    total_list_x = []; total_list_y = []
    for sub in [args.test_subj]:
        data_name = "subj" + str(sub).zfill(2) + ".npz"
        o_list = np.load(args.path + data_name)
        data_len = o_list['x'].shape[0]
        temp_x = o_list['x'][int(data_len/2):]
        temp_y = o_list['y'][int(data_len/2):]
        total_list_x.append(temp_x); total_list_y.append(temp_y)
    data_x_test, data_y_test = np.vstack(total_list_x), np.vstack(total_list_y)
    '''
    #### make train test
    # test_train_split = 0.5
    # dataset= data[subject_id]
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # test_split = int(np.floor(test_train_split * dataset_size))
    # train_indices, test_indices = indices[:test_split], indices[test_split:]
    # np.random.shuffle(train_indices)

    # train_set1 = BcicDataset_window(torch.utils.data.Subset(dataset, indices=train_indices), window_num=0)
    # train_set2 = BcicDataset_window(torch.utils.data.Subset(dataset, indices=train_indices), window_num=1)
    # train_set = ConcatDataset([train_set1, train_set2])
    # test_set = torch.utils.data.Subset(dataset, indices=test_indices)

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    train_data = data_loader_bcic.Dataset(args, phase="train")
    test_data = data_loader_bcic.Dataset(args, phase="valid")
    
    train_set1 = BcicDataset_window(train_data, window_num=0)
    train_set2 = BcicDataset_window(train_data, window_num=1)
    train_set = ConcatDataset([train_set1, train_set2])
    
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    crop_size =1000
    model = ShallowConvNet_dk(n_classes, 22, crop_size)
    
    # print(model)
    cuda = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        model.cuda(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch-1)

    results_columns = ['test_loss', 'test_accuracy']
    df = pd.DataFrame(columns=results_columns)
    path = "./output_bcic_modi"
    best_acc = 0
    for epochidx in range(1, args.epoch):
        print(epochidx)
        train(10, model, device, train_loader,optimizer,scheduler,cuda, 0)
        test_loss, test_score = eval(model, device, test_loader)
        results = {'test_loss': test_loss, 'test_accuracy': test_score}
        df = df.append(results, ignore_index=True)
        print(results)

        if test_score >= best_acc:
            best_acc = test_score
            torch.save(model.state_dict(), os.path.join(
                path, "model_subject{}_best.pt".format(args.test_subj)))


    best_model = ShallowConvNet_dk(n_classes, 22, crop_size)
    # best_model = EEGNet(n_classes, 22, crop_size)
    best_model.load_state_dict(torch.load(os.path.join(
        path,
        "model_subject{}_best.pt").format(args.test_subj), map_location=device))
    if cuda:
        best_model.cuda(device=device)

    print("best accuracy")
    value1, value2 = eval(best_model, device, test_loader)
    print(f"==={value1, value2}===")

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