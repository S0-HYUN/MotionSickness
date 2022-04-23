import argparse
import os
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *
from sklearn.preprocessing import MinMaxScaler
import re

class Load_Data() :
    def __init__(self, args) :
        datalist = os.listdir(args.data_path)
        datalist = sorted(datalist)
        ch_num = args.channel_num + 1

        for i in range(len(datalist)) :
            numbers = re.sub(r'[^0-9]', '', datalist[i]) # Extract only numbers
            subj = int(numbers[:2]); day = int(numbers[2]); expt = int(numbers[3])

            print(".....[" + datalist[i] + "] loading.....")
            o_data = np.array(pd.read_csv(args.data_path + datalist[i], sep=',', header=0))
            o_data_f = pd.DataFrame(o_data)
            naming(o_data_f)
            o_data_f.drop(['Time', 'ECG.(uV)', 'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)'], axis =  1, inplace = True) # 지우는 열 -> time도 포함시켜야 하는지 고민. 우선 data_loader.py에서 Time이용해서 조절하기 때문에 냅둠.
            
            #---# downsampling -> 250 #---#
            o_data_f = o_data_f.assign(g = o_data_f.index % 2).query('g==0')    # 0으로 할건지 1로 할 건지인데, 0으로 해야 시작시간이 바르기 때무네..
            o_data_f.drop(['g'], axis = 1, inplace = True)                      # g열 버림
            o_data_f.reset_index(drop = True, inplace = True)                   # 인덱스 재설정 0부터~~

            #---# visualization of each channel (check for trend) #---#
            # from matplotlib import pyplot as plt
            # plt.plot(np.arange(len(o_data_f.iloc[:,1])), o_data_f.iloc[:,1])
            # plt.ylim(-1000, 1000)
            # plt.savefig("./plots/202111300521_channel_minmax_check_column1_.png")

            #---# changing scale and to_numpy #---#
            o_data_f.iloc[:,:-1] = o_data_f.iloc[:,:-1].apply(lambda x : x / 100)

            o_data = o_data_f.to_numpy()            # 다시 numpy 배열로
            cut = len(o_data) % args.one_bundle          # 나머지 부분 잘라내기 cut = len(o_data) % (one_bundle * ch_num) 여기 꼭 확인

            o_data = o_data[:-int(cut)].reshape(-1, args.one_bundle, ch_num)
            
            #---# bandpass filtering #---#
            b, t, c = o_data.shape
            o_data = mne.filter.filter_data(o_data.reshape(-1, ch_num).T, sfreq = 250, l_freq = args.lower_freq, h_freq = args.high_freq, picks = np.s_[1:-2]).T.reshape(b, t, c)
            
            x = o_data[:,:,:-1]; y = o_data[:,:,-1] # x.shape -> (512, 750, 28)
 
            #---# Use of MinMaxScaler #---# -> 이런 거 하지말래
            # scalar = MinMaxScaler(feature_range=(-10,10))
            # x = x.reshape(-1,28)
            # x = scalar.fit_transform(x)
            # x = x.reshape(-1, args.one_bundle, ch_num-1)

            #--------------------------#
            #---# save single data #---#  
            #--------------------------#
            output_dir = str(args.output_path) + "/Single/Class" + str(args.class_num) + "/Expt" + str(expt) + "/day" + str(day)
            filename = "subj" + str(subj).zfill(2)
            file_path_name = output_dir + "/" + filename + ".npz"
            check_and_save(output_dir, filename, x, y, subj)

            #-------------------------#
            #---# save split data #---#
            #-------------------------#
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.test_size, shuffle=True, random_state=1004)
            output_dir = str(args.output_path) + "/Split" + str(args.test_size) + "/Class" + str(args.class_num) + "/Expt" + str(expt) + "/day" + str(day)
            
            filename = "subj" + str(subj).zfill(2) + "_train"
            check_and_save(output_dir, filename, x_train, y_train, subj)
            
            filename = "subj" + str(subj).zfill(2) + "_val"
            check_and_save(output_dir, filename, x_val, y_val, subj)

    def __getitem__(self) :
        return

def check_and_save(out_dir, filename, x, y, subj):
    if not os.path.isfile(filename):
        save_npzfile(out_dir, filename, x, y, subj)
    else:
        print(f"{filename} is already exist!")

def save_npzfile(out_dir, filename, x, y, subj) :
    create_folder(out_dir)
    save_dict = {
            "x" : x,
            "y" : y,
            "subj" : subj
    }
    np.savez(os.path.join(out_dir, filename), **save_dict)


def naming(df):
        df.columns = ['Time', 'Fp1(uV)', 'Fp2(uV)', 'AF3(uV)','AF4(uV)', 'F7(uV)', 
            'F8(uV)', 'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
            'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)',	'P8(uV)', 'P3(uV)',	'Pz(uV)',
            'P4(uV)', 'PO7(uV)', 'PO8(uV)',	'PO3(uV)', 'PO4(uV)', 'O1(uV)',	'O2(uV)', 'ECG.(uV)',
            'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)', 'TRIGGER(DIGITAL)']

def main() :
    parser = argparse.ArgumentParser(); 
    parser.add_argument("--test_size", type=float, default=0.1); 
    parser.add_argument("--lower_freq", type=float, default=0.5)
    parser.add_argument("--high_freq", type=float, default=50); 
    parser.add_argument("--score_list", type=list, default=[1,6]); #[0,4] # [3] -> 0,1,2,3 /4,5,6,7,8,9     # [3,7] -> 0,1,2,3 / 4,5,6 / 7,8,9    # [1,6] -> 0,1 / 2,3,4,5 / 6,7,8,9
    parser.add_argument("--data_path", type=str, default='/opt/workspace/MS_DATA/rest/')
    parser.add_argument("--channel_num", type=int, default=28)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--one_bundle", type=int, default=int(1500/2)) # 500hz -> 3초에 1500행
    parser.add_argument("--output_path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_rest_include_subN')
    args = parser.parse_args()
    Load_Data(args)
	
if __name__ == "__main__" :
    main()