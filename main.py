from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader
from prepare_MS import createFolder
import trainer
import os
import pandas as pd
import Network.EEGNet_models
import Network.DeepConvNet_models
import argparse
import subprocess
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(); 
    # parser.add_argument("--path", type=str, default='/opt/workspace/soxo/Motionsickness_Data/PREPROCESSED_DATA/')
    parser.add_argument("--path", type=str, default='/opt/workspace/soxo/output/')
    parser.add_argument("--lr_list", type=list, default=[1e-5, 1e-4, 1e-3])     #[1e-5, 1e-4, 1e-3]
    parser.add_argument("--wd_list", type=list, default=[1e-5, 1e-4, 1e-3])     #[1e-5, 1e-4, 1e-3]
    parser.add_argument("--batch_size", type=int, default=512)              #512
    parser.add_argument("--epoch", type=int, default=100)                   #100
    parser.add_argument("--one_bundle", type=int, default=int(1500/2))
    parser.add_argument("--channel_num", type=int, default=28)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--expt", type=int, default="1")  # 1:오전 2:오후
    # parser.add_argument("--val_list", type=list, default=[2]) # 필요없어보임
    parser.add_argument("--test_subj", type=int, default=13)
    parser.add_argument("--test_size", type=float, default=0.5); 
    parser.add_argument("--param_path", type=str, default="/opt/workspace/soxo/soh_code_f/param")
    parser.add_argument("--runs_path", type=str, default="/opt/workspace/soxo/soh_code_f/runs")
    args = parser.parse_args()

    return args


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

    if not os.path.isdir(args.path + "Split" + str(args.test_size)) :
        process = subprocess.Popen(["python", "/opt/workspace/soxo/soh_code_f/prepare_MS.py", "-test_size", str(args.test_size)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.communicate()

    # test_file = filing.calc_test_file(test_subj, expt)
    #   val_list = filing.calc_test_file(val_subj, expt)
    # train_list.remove(test_file)

    info = {} # information dictionary
    info['class'] = args.class_num
    info['expt'] = args.expt
    info['test_size'] =  args.test_size
    info['one_bundle'] = args.one_bundle
    info['channel_num'] = args.channel_num

    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'acc', 'f1', 'loss'])
    idx = 0
    
    test_list = [args.test_subj]
    train_list = list(range(1, 24))
    # train_list = [x for x in train_list if x not in (args.val_list + args.test_list)] 
    train_list = [x for x in train_list if x not in (test_list)] 

    data = data_loader.Dataset(info, args.path, train_list, test_list, data_type = 'train')
    data_valid = data_loader.Dataset(info, args.path, None, test_list, data_type = 'valid')
    data_test = data_loader.Dataset(info, args.path, train_list, test_list, data_type = 'test')

    model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device = device)
    # model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device = device)
    # model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)    

    lr_list = [1e-5, 1e-4, 1e-3]
    wd_list = [1e-5, 1e-4, 1e-3]

    prepare_folder(args.param_path, args.runs_path)

    for lr in lr_list :
        for wd in wd_list :
            print("\n====================================lr{}, wd{}====================================".format(lr, wd))

            trainer1 = trainer.trainer(info, model, args.epoch, lr, optimizer = 'Adam', weight_decay=wd)
            # acc, f1, cm, loss = trainer1.only_train(data, batch_size=args.batch_size)
            acc, f1, cm, loss = trainer1.train(data, data_valid, args.batch_size)
            last_param = os.listdir('./param/lr{}_wd{}/'.format(lr, wd))[-1] ############################################# 저장 위치
            best_model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device=device)
            # best_model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device=device)
            # best_model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)
            best_model.load_state_dict(torch.load('./param/lr{}_wd{}/'.format(lr, wd)+last_param)) 

            trainer2 = trainer.trainer(info, best_model, 1, 0.1, 'Adam', 0.1)
            acc_v, f1_v, cm_v, loss_v = trainer2.prediction(data_test, args.batch_size, test=True)

            df.loc[idx] = [args.test_subj, lr, wd, acc_v, f1_v, loss_v.cpu().numpy()]
 
            idx += 1 
            df.to_excel('results_EEGNET_subj{}.xlsx'.format(args.test_subj), header = True, index = False)

def prepare_folder(param_path, runs_path) :
    if os.path.isdir(param_path) :
        print("Remove folder")
        shutil.rmtree(param_path)
    if os.path.isdir(runs_path) :
        shutil.rmtree(runs_path)
    createFolder(param_path) 

if __name__ == "__main__" :
    args = parse_arguments()
    main(args)

