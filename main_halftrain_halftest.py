from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader
import trainer
import os
import pandas as pd
import Network.EEGNet_models
import Network.DeepConvNet_models
import argparse
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(); 
    # parser.add_argument("--path", type=str, default='/opt/workspace/soxo/Motionsickness_Data/PREPROCESSED_DATA/')
    parser.add_argument("--path", type=str, default='/opt/workspace/soxo/output/')
    parser.add_argument("--lr_list", type=list, default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--wd_list", type=list, default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--one_bundle", type=int, default=int(1500/2))
    parser.add_argument("--channel_num", type=int, default=28)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--expt", type=int, default="1")  # 1:오전 2:오후
    parser.add_argument("--val_list", type=list, default=[2])
    parser.add_argument("--test_list", type=list, default=[13])
    parser.add_argument("--test_size", type=float, default=0.5); 
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

    # train_list = list(range(1, 24))
    # train_list = [x for x in train_list if x not in (args.val_list + args.test_list)] 


    for lr in args.lr_list :
        for wd in args.wd_list :
            for subj_num in range(1, 24) :
                train_list = []
                val_list = [subj_num]
            
                data = data_loader.Dataset(info, args.path, train_list, val_list, data_type = 'train') # args.val_list
                data_validation = data_loader.Dataset(info, args.path, None, val_list, data_type = 'valid') # args.val_list

                # model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device = device)
                # model = Network.DeepConvNet_models.ShallowConvNet_dk(args.class_num, args.channel_num).to(device = device)
                model = Network.DeepConvNet_models.DeepConvNet(args.class_num, args.channel_num, args.one_bundle).to(device = device)   
                print("\n====================================lr{}, wd{}====================================".format(lr, wd))

                trainer1 = trainer.trainer(info, model, args.epoch, lr, optimizer = 'Adam', weight_decay=wd)
                acc, f1, cm, loss = trainer1.only_train(data, batch_size=args.batch_size)

                last_param = os.listdir('./param/lr{}_wd{}/'.format(lr, wd))[-1] ############################################# 저장 위치
                # best_model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device=device)
                # best_model = Network.DeepConvNet_models.ShallowConvNet_dk(args.class_num, args.channel_num).to(device=device)
                best_model = Network.DeepConvNet_models.DeepConvNet(args.class_num, args.channel_num, args.one_bundle).to(device = device)
                best_model.load_state_dict(torch.load('./param/lr{}_wd{}/'.format(lr, wd)+last_param)) 

                trainer2 = trainer.trainer(info, best_model, 1, 0.1, 'Adam', 0.1)
                acc_v, f1_v, cm_v, loss_v = trainer2.prediction(data_validation, args.batch_size, test=True)

                df.loc[idx] = [subj_num, lr, wd, acc_v, f1_v, loss_v.cpu().numpy()]

                idx += 1 
                df.to_excel('results.xlsx', header = True, index = False)

if __name__ == "__main__" :
    args = parse_arguments()
    main(args)


'''
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

    df = pd.DataFrame(columns = ['test_subj', 'expt', 'lr', 'wd', 'acc', 'f1', 'loss'])
    idx = 0

    train_list = list(range(1, 24))
    train_list = [x for x in train_list if x not in (args.val_list + args.test_list)] 

    data = data_loader.Dataset(info, args.path, train_list, args.val_list, data_type = 'train')
    data_validation = data_loader.Dataset(info, args.path, None, args.val_list, data_type = 'valid')

    model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device = device)
    # model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device = device)
    # model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)    

    lr_list = [1e-5, 1e-4, 1e-3]
    wd_list = [1e-5, 1e-4, 1e-3]

    for lr in lr_list :
        for wd in wd_list :
            print("\n====================================lr{}, wd{}====================================".format(lr, wd))

            trainer1 = trainer.trainer(model, args.epoch, lr, optimizer = 'Adam', weight_decay=wd)
            acc, f1, cm, loss = trainer1.only_train(data, batch_size=args.batch_size)

            last_param = os.listdir('./param/lr{}_wd{}/'.format(lr, wd))[-1] ############################################# 저장 위치
            best_model = Network.EEGNet_models.EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device=device)
            # best_model = Network.DeepConvNet_models.ShallowConvNet_dk(class_num, channel_num).to(device=device)
            # best_model = Network.DeepConvNet_models.DeepConvNet(class_num, channel_num, one_bundle).to(device = device)
            best_model.load_state_dict(torch.load('./param/lr{}_wd{}/'.format(lr, wd)+last_param)) 

            trainer2 = trainer.trainer(best_model, 1, 0.1, 'Adam', 0.1)
            acc_v, f1_v, cm_v, loss_v = trainer2.prediction(data_validation, args.batch_size, test=True)

            df.loc[idx] = [train_list[0], expt, lr, wd, acc_v, f1_v, loss_v.cpu().numpy()]

            idx += 1 
            df.to_excel('results.xlsx', header = True, index = False)
'''