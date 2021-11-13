import argparse
import datetime
from random import choice

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

        #---# Model #---#
        parser.add_argument("--model", type=str)
        

        #---# Path #---#
        parser.add_argument("--path", type=str, default='/opt/workspace/xohyun/MS/output/')
        parser.add_argument("--param_path", type=str, default="/opt/workspace/xohyun/MS/param")
        parser.add_argument("--runs_path", type=str, default="/opt/workspace/xohyun/MS/runs")

        #---# Train #---#

        # Try several things at once
        # parser.add_argument("--lr_list", type=list, default=[1e-5, 1e-4, 1e-3])     #[1e-5, 1e-4, 1e-3]
        # parser.add_argument("--wd_list", type=list, default=[1e-5, 1e-4, 1e-3])     #[1e-5, 1e-4, 1e-3]
        
        parser.add_argument("--lr_list", type=float, default=1e-4)
        parser.add_argument("--wd_list", type=float, default=1e-4)     

        parser.add_argument("--criterioin", type=str, default="CEE")
        parser.add_argument("--optimizer", type=str, default="Adam")

        parser.add_argument("--batch_size", type=int, default=512)              #512
        parser.add_argument("--epoch", type=int, default=100)                   #100
        parser.add_argument("--one_bundle", type=int, default=int(1500/2))
        parser.add_argument("--channel_num", type=int, default=28)
        parser.add_argument("--class_num", type=int, default=3)
        parser.add_argument("--expt", type=int, default="1", help="1:오전,2:오후")  
        parser.add_argument("--test_subj", type=int, default=13)
        parser.add_argument("--remove_subj", type=list, default=[])
        parser.add_argument("--test_size", type=float, default=0.5); 
        
        #---# Device #---#
        parser.add_argument('--device', default=0, help="cpu or gpu number")
        args = parser.parse_args()

        return args
