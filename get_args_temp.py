import os
import argparse
import datetime
from random import choice
from utils import create_folder, prepare_folder

class Args:
    def __init__(self):
        self.args = self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(); 

        #---# Time #---#
        now = datetime.datetime.now()
        parser.add_argument('--date', default=now.strftime('%Y-%m-%d'), help="Please do not enter any value.")
        parser.add_argument('--time', default=now.strftime('%H:%M:%S'), help="Please do not enter any value.")

        #---# Mode #---#
        parser.add_argument("--mode", default="train", choices=["train", "test"])
        # parser.add_argument("--mode", default="train", choices=["train", "test"])
        parser.add_argument("--seed", default=1004, type=int)
        parser.add_argument("--DA", default=False, type=bool)
        if parser.parse_known_args()[0].DA == True:
            parser.add_argument('--da_epoch', type=int, default=100, required=True)
            parser.add_argument('--da_lr', type=float, default=1e-4, required=True)

        #---# Model #---#
        parser.add_argument("--model", type=str, default="DeepConvNet", choices=['DeepConvNet', 'ShallowConvNet', 'EEGNet', 'CRL', 'soso', 'ODML']) #DeepConvNet, ShallowConvNet, EEGNet, CRL

        #---# Path #---# ###### 여기에 안쓰이는 거 있는지 확인
        ### Motion sickness
        # parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_scale_0_123_456789/')
        # parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_scale_01_2345_6789/')
        
        ### bci competition
        # parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS_codes/output_bcic_modi/')
        parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_scale_01_2345_6789_include_subN/') #Files_scale_01_2345_6789+rest
        parser.add_argument("--param_path", type=str, default="/opt/workspace/xohyun/MS_codes/param")
        parser.add_argument("--runs_path", type=str, default="/opt/workspace/xohyun/MS_codes/runs")
        parser.add_argument("--save_path", type=str, default="/opt/workspace/xohyun/MS_codes/train/")
        parser.add_argument("--save_folder", type=str, default="/opt/workspace/xohyun/MS_codes/train/")
        parser.add_argument("--save_pastfolder", type=str, default="/opt/workspace/xohyun/MS_codes/train_PAST")
        parser.add_argument("--load_path", type=str, default="/opt/workspace/xohyun/MS_codes/train/")
        parser.add_argument("--ft_folder", type=str, default="/opt/workspace/xohyun/MS_codes/train_da/") # for fine-tuning

        #---# Train #---#

        # Try several things at once
        # parser.add_argument("--lr_list", type=list, default=[1e-5, 1e-4, 1e-3]) # [1e-5, 1e-4, 1e-3]
        # parser.add_argument("--wd_list", type=list, default=[1e-5, 1e-4, 1e-3]) # [1e-5, 1e-4, 1e-3]
        parser.add_argument("--n_queries", type=int, default=150)
        parser.add_argument("--standard", type=str, default="acc", choices=['loss', 'acc', 'f1'])

        parser.add_argument('--scheduler', '-sch')
        if parser.parse_known_args()[0].scheduler == 'exp':
            parser.add_argument('--gamma', type=float, required=True)
        elif parser.parse_known_args()[0].scheduler == 'step':
            parser.add_argument('--step_size', type=int, required=True, default=10)
            parser.add_argument('--gamma', type=float, required=True, default=0.5)
        elif parser.parse_known_args()[0].scheduler == 'multi_step':
            parser.add_argument('--milestones', required=True) # type=str2list_int
            parser.add_argument('--gamma', type=float, required=True)
        elif parser.parse_known_args()[0].scheduler == 'plateau':
            parser.add_argument('--factor', type=float, required=True)
            parser.add_argument('--patience', type=int, required=True)
        elif parser.parse_known_args()[0].scheduler == 'cosine':
            parser.add_argument('--T_max', type=float, help='Max iteration number', default=50)
            parser.add_argument('--eta_min', type=float, help='minimum learning rate', default=0)
        elif parser.parse_known_args()[0].scheduler == 'one_cycle': #default로 cosine이라 anneal_strategy를 따로 변수로 안잡음
            parser.add_argument('--max_lr', type=float, default=0.1)
            parser.add_argument('--steps_per_epoch', type=int, default=10) # 증가하는 cycle의 반
            parser.add_argument('--cycle_epochs', type=int, default=10)

        parser.add_argument("--criterion", type=str, default="CEE")
        parser.add_argument("--optimizer", type=str, default="AdamW")   # AdamW, SGD

        parser.add_argument("--metrics", type=list, default=["loss", "acc"])

        parser.add_argument("--lr", type=float, default=0.001) # 1e-3 #bcic:0.000625
        parser.add_argument("--wd", type=float, default=0.0001) # 1e-3

        parser.add_argument("--batch_size", type=int, default=256)      # 512
        parser.add_argument("--epoch", type=int, default=100)          # 3000
        parser.add_argument("--one_bundle", type=int, default=750)     # int(1500/2) / 1125
        parser.add_argument("--channel_num", type=int, default=28)      # 28 / 22
        parser.add_argument("--class_num", type=int, default=3)
        parser.add_argument("--expt", type=int, default=1, help="1:오전,2:오후")
        if parser.parse_known_args()[0].expt == 1:
            parser.add_argument("--remove_subj", type=list, default=[]) #1,2,4,14,16,17,19
        else:
            parser.add_argument("--remove_subj", type=list, default=[4,8,11,17]) 
        parser.add_argument("--test_subj", type=int, default=12)
        parser.add_argument("--test_size", type=float, default=0.1); # 0.05
        # parser.add_argument("-")
    
        #---# Device #---#
        parser.add_argument("--device", default=3, help="cpu or gpu number")

        args = parser.parse_args()
        return args

    def set_save_path(self):
        if len(os.listdir(self.args.save_pastfolder)) == 23: #if os.path.isdir(self.args.save_folder) :
            if not os.path.isdir(self.args.save_pastfolder) :
               create_folder(self.args.save_pastfolder)
    
            import shutil
            des = os.path.join(self.args.save_pastfolder, str(len(os.listdir(self.args.save_pastfolder))+1)); create_folder(des)
            shutil.move(self.args.save_folder, des)
        
        save_dir = os.path.join(self.args.save_folder, str(self.args.test_subj)); create_folder(save_dir)
        self.args.save_path = os.path.join(save_dir, str(len(os.listdir(save_dir))+1)); create_folder(self.args.save_path)
        print(f"=== save_path === [{self.args.save_path}]")
        # create_folder(self.args.save_folder)
        # save_dir = os.path.join(self.args.save_folder, str(len(os.listdir(self.args.save_folder))+1)); create_folder(save_dir)
        # self.args.save_path = os.path.join(save_dir, str(self.args.test_subj)); create_folder(self.args.save_path)
        # print(f"=== save_path === [{self.args.save_path}]")

    def set_save_path_DA(self):
        save_dir = os.path.join(self.args.ft_folder, str(self.args.test_subj)); create_folder(save_dir)
        self.args.save_path = os.path.join(save_dir, str(len(os.listdir(save_dir))+1)); create_folder(self.args.save_path)
        print(f"=== save_path for DA === [{self.args.save_path}]")
        # create_folder(self.args.ft_folder)
        # save_dir = os.path.join(self.args.ft_folder, str(len(os.listdir(self.args.ft_folder))+1)); create_folder(save_dir)
        # self.args.save_path = os.path.join(save_dir, str(self.args.test_subj)); create_folder(self.args.save_path)
        # print(f"=== save_path for DA === [{self.args.save_path}]")

    def get_load_path(self):
        save_pth = os.path.join(self.args.load_path, str(self.args.test_subj))
        self.args.load_path = os.path.join(save_pth, str(len(os.listdir(save_pth))))
        print(f"+++ load_path : [{self.args.load_path}] +++")
        # save_pth = os.path.join(self.args.load_path, str(len(os.listdir(self.args.load_path))))
        # self.args.load_path = os.path.join(save_pth, str(self.args.test_subj))
        # print(f"+++ load_path : [{self.args.load_path}] +++")
    
    def get_load_path_DA(self):
        save_pth = os.path.join(self.args.ft_folder, str(self.args.test_subj))
        self.args.load_path = os.path.join(save_pth, str(len(os.listdir(save_pth))))
        print(f"+++ load_path : [{self.args.load_path}] +++")
        # save_pth = os.path.join(self.args.ft_folder, str(len(os.listdir(self.args.ft_folder))))
        # self.args.load_path = os.path.join(save_pth, str(self.args.test_subj))
        # print(f"+++ load_path : [{self.args.load_path}] +++")