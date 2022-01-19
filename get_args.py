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

        #---# Model #---#
        parser.add_argument("--model", type=str, default="ShallowConvNet") #DeepConvNet, ShallowConvNet, EEGNet

        #---# Path #---# ###### 여기에 안쓰이는 거 있는지 확인
        ### Motion sickness
        # parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_scale_0_123_456789/')
        # parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_scale_01_2345_6789/')
        
        ### bci competition
        parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS_codes/output_bcic_modi/')

        parser.add_argument("--param_path", type=str, default="/opt/workspace/xohyun/MS_codes/param")
        parser.add_argument("--runs_path", type=str, default="/opt/workspace/xohyun/MS_codes/runs")
        parser.add_argument("--save_path", type=str, default="/opt/workspace/xohyun/MS_codes/train/")
        parser.add_argument("--save_folder", type=str, default="/opt/workspace/xohyun/MS_codes/train/")
        parser.add_argument("--load_path", type=str, default="/opt/workspace/xohyun/MS_codes/train/")

        #---# Train #---#

        # Try several things at once
        # parser.add_argument("--lr_list", type=list, default=[1e-5, 1e-4, 1e-3]) # [1e-5, 1e-4, 1e-3]
        # parser.add_argument("--wd_list", type=list, default=[1e-5, 1e-4, 1e-3]) # [1e-5, 1e-4, 1e-3]
        parser.add_argument("--n_queries", type=int, default=150)

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

        parser.add_argument("--criterion", type=str, default="CEE")
        parser.add_argument("--optimizer", type=str, default="AdamW")   # AdamW

        parser.add_argument("--metrics", type=list, default=["loss", "acc"])

        parser.add_argument("--lr", type=float, default=0.000625) # 1e-3
        parser.add_argument("--wd", type=float, default=0) # 1e-3

        parser.add_argument("--batch_size", type=int, default=8)      # 512
        parser.add_argument("--epoch", type=int, default=100)          # 3000
        parser.add_argument("--one_bundle", type=int, default=1125)     # int(1500/2) / 1125
        parser.add_argument("--channel_num", type=int, default=22)      # 28 / 22
        parser.add_argument("--class_num", type=int, default=4)
        parser.add_argument("--expt", type=int, default=1, help="1:오전,2:오후")
        if parser.parse_known_args()[0].expt == 1:
            parser.add_argument("--remove_subj", type=list, default=[1,2,4,14,16,17,19])
        else:
            parser.add_argument("--remove_subj", type=list, default=[4,8,11,17]) 
        parser.add_argument("--test_subj", type=int, default=7)
        parser.add_argument("--test_size", type=float, default=0.5); # 0.05
        # parser.add_argument("-")
    
        #---# Device #---#
        parser.add_argument('--device', default=2, help="cpu or gpu number")

        args = parser.parse_args()
        return args

    def set_save_path(self):
        save_dir = os.path.join(self.args.save_folder, str(self.args.test_subj))
        create_folder(save_dir)
        
        if self.args.mode == "train":
            self.args.save_path = os.path.join(save_dir, str(len(os.listdir(save_dir))+1))
            create_folder(self.args.save_path)
            print(f"=== save_path : [{self.args.save_path}] ===")

    def get_load_path(self):
        save_pth = os.path.join(self.args.load_path, str(self.args.test_subj))
        self.args.load_path = os.path.join(save_pth, str(len(os.listdir(save_pth))))
