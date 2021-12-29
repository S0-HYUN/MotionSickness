import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.preprocessing import MinMaxScaler

class Load_Data() :
    def __init__(self, args) :
        o_data = np.array(pd.read_csv("/opt/workspace/xohyun/MS_copy/testdata.csv", sep=',', header=0))
        o_data_f = pd.DataFrame(o_data)
        print(o_data_f.shape)
        #---# changing scale and to_numpy #---#
        # o_data_f.iloc[:,:-1] = o_data_f.iloc[:,:-1].apply(lambda x : x / 100)
        naming(o_data_f)
        o_data_f.drop(['Time', 'ECG.(uV)', 'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)'], axis =  1, inplace = True) # 지우는 열 -> time도 포함시켜야 하는지 고민. 우선 data_loader.py에서 Time이용해서 조절하기 때문에 냅둠.
        print(o_data_f.shape)

        o_data = o_data_f.to_numpy()   # 다시 numpy 배열로
        cut = len(o_data) % 5          # 나머지 부분 잘라내기 cut = len(o_data) % (one_bundle * ch_num) 여기 꼭 확인
        ch_num = 29
        o_data = o_data[:-int(cut)].reshape(-1, 5, ch_num)
        
        x = o_data[:,:,:-1];    y = o_data[:,:,-1] # x.shape -> (512, 750, 28)

        #---# Use of MinMaxScaler #---#
        # scalar = MinMaxScaler(feature_range=(-10,10))
        # x = x.reshape(-1,28)
        # x = scalar.fit_transform(x)
        # x = x.reshape(-1, args.one_bundle, ch_num-1)

def naming(df):
        df.columns = ['Time', 'Fp1(uV)', 'Fp2(uV)', 'AF3(uV)','AF4(uV)', 'F7(uV)', 
            'F8(uV)', 'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
            'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)',	'P8(uV)', 'P3(uV)',	'Pz(uV)',
            'P4(uV)', 'PO7(uV)', 'PO8(uV)',	'PO3(uV)', 'PO4(uV)', 'O1(uV)',	'O2(uV)', 'ECG.(uV)',
            'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)', 'TRIGGER(DIGITAL)']

def save_npzfile(out_dir, filename, x, y) :
    createFolder(out_dir)
    save_dict = {
            "x" : x,
            "y" : y,
    }
    np.savez(os.path.join(out_dir, filename), **save_dict)

def main() :
    parser = argparse.ArgumentParser(); 
    parser.add_argument("--test_size", type=float, default=0.5); 
    parser.add_argument("--lower_freq", type=float, default=0.5)
    parser.add_argument("--high_freq", type=float, default=50); 
    parser.add_argument("--score_list", type=list, default=[1,6]);  # [3] -> 0,1,2,3 /4,5,6,7,8,9     # [3,7] -> 0,1,2,3 / 4,5,6 / 7,8,9    # [1,6] -> 0,1 / 2,3,4,5 / 6,7,8,9
    parser.add_argument("--data_path", type=str, default='/opt/workspace/MS_DATA/PREPROCESSED_DATA/')
    parser.add_argument("--channel_num", type=int, default=28)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--one_bundle", type=int, default=int(1500/2)) # 500hz -> 3초에 1500행
    parser.add_argument("--output_path", type=str, default='/opt/workspace/xohyun/MS/Files_scale')
    args = parser.parse_args()
    Load_Data(args)

if __name__ == "__main__" :
    main()