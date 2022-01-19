import argparse
import os
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Load_Data() :
    def __init__(self, args) :
        datalist = os.listdir(args.data_path)
        datalist = sorted(datalist)

        ch_num = args.channel_num + 1
        
        for i in range(len(datalist)) : 
            if i % 2 == 0 : expt = 1 
            else : expt = 2 

            if (i % 4 == 0) or (i % 4 == 1) : day = 1
            else : day = 2
            subj = (i // 4) + 1
            
            print(".....[" + datalist[i] + "] loading.....")
            o_data = np.array(pd.read_csv(args.data_path + datalist[i], sep=',', header=0))
            o_data_f = pd.DataFrame(o_data)
            naming(o_data_f)
            o_data_f.drop(['Time', 'ECG.(uV)', 'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)'], axis =  1, inplace = True) # 지우는 열 -> time도 포함시켜야 하는지 고민. 우선 data_loader.py에서 Time이용해서 조절하기 때문에 냅둠.
            
            #---# set class #---#
            if args.class_num == 2:
                o_data_f['TRIGGER(DIGITAL)'] = o_data_f['TRIGGER(DIGITAL)'].apply(lambda x:0 if x <= args.score_list[0] else 1)
            elif args.class_num == 3:
                o_data_f['TRIGGER(DIGITAL)'] = o_data_f['TRIGGER(DIGITAL)'].apply(lambda x:0 if x <= args.score_list[0] else (2 if x >= args.score_list[1] else 1))

            
            o_data_f_only_eeg = o_data_f.copy()
            o_data_f_only_trigger = o_data_f_only_eeg['TRIGGER(DIGITAL)']
            o_data_f_only_trigger = o_data_f_only_trigger.to_frame()
            # print(o_data_f_only_trigger)
            # o_data_f_only_trigger = o_data_f_only_trigger.assign(g = o_data_f_only_trigger.index % 3).query('g==0')
            # o_data_f_only_trigger.drop(['g'], axis = 1, inplace = True)
            # # o_data_f_only_trigger.reset_index(drop = True, inplace = True) 
            # print(o_data_f_only_trigger)

            # o_data_only_trigger = o_data_f_only_trigger.to_numpy()
            # info_trigger = mne.create_info(ch_names=['trigger'], sfreq=50)
            # raw_trigger = mne.io.RawArray(o_data_only_trigger.transpose(), info_trigger)
            # print(raw_trigger)
            # # mne.make_fixed_length_events(o_data_f_only_trigger)
            # events = mne.find_events(raw_trigger, stim_channel='trigger') 
            # print(events)

            
            # tt = o_data_f_only_trigger.index % 3 != 0
            # print("===", tt)
            # total_num = len(o_data_f_only_trigger)
            # total_range = list(range(10))
            # print(total_range[total_range % 3 == 0])
            # kk = total_range[tt]
            # print("+=+", kk); 
            
            o_data_f_only_trigger.loc[o_data_f_only_trigger.index % 3 != 0,:] = 0
            print(o_data_f_only_trigger)
            o_data_f_only_eeg.drop(['TRIGGER(DIGITAL)'], axis = 1, inplace = True)

            
            # o_data_f_only_trigger.loc[,:]
            
            o_data_only_trigger = o_data_f_only_trigger.to_numpy()
            print(o_data_only_trigger)
            
            ch_names = ['Fp1', 'Fp2', 'AF3','AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8',
            'C3', 'C4', 'CP5', 'CP6', 'P7',	'P8', 'P3',	'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1', 'O2']
            info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types='eeg')
            montage = mne.channels.make_standard_montage("standard_1005")

            o_data_only_eeg = o_data_f_only_eeg.to_numpy()
            print(o_data_only_eeg)
            o_data_only_eeg = o_data_only_eeg * (1e-6)
            print("=============")
            print(o_data_only_eeg)
            raw = mne.io.RawArray(o_data_only_eeg.transpose(), info)
            raw.filter(l_freq=args.lower_freq, h_freq=args.high_freq)
            
            new_events = mne.make_fixed_length_events(raw, start=0, stop = len(raw), duration=0.03)
            print(new_events)
            raise
            # figure
            raw.plot()
            plt.savefig("test2.png")
            o_data_f_only_label = o_data_f['TRIGGER(DIGITAL)']
            o_data_only_label = o_data_f_only_label.to_numpy().astype(np.int64)

            # print(o_data_only_label)
            # o_data_only_label = o_data_only_label.astype(np.int64)
            # print(o_data_only_label)
            # events = mne.find_events(raw, stim_channel=ch_names)
            # events = mne.event.define_target_events(o_data_only_label)
            # events = mne.read_events(o_data_only_label)
            events = mne.make_fixed_length_events(raw, duration=1) #start=0, stop=2000, 
            epoch = mne.Epochs(raw, events)
            epoch._data = epoch.get_data()

            # mne.Epochs(raw, o_data_only_label)

            raise
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
            
            x = o_data[:,:,:-1];    y = o_data[:,:,-1] # x.shape -> (512, 750, 28)
     
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
            check_and_save(output_dir, filename, x, y)
            
            #-------------------------#
            #---# save split data #---#
            #-------------------------#
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.test_size, shuffle=True, random_state=1004)
            output_dir = str(args.output_path) + "/Split" + str(args.test_size) + "/Class" + str(args.class_num) + "/Expt" + str(expt) + "/day" + str(day)
            
            filename = "subj" + str(subj).zfill(2) + "_train"
            check_and_save(output_dir, filename, x_train, y_train)
            
            filename = "subj" + str(subj).zfill(2) + "_val"
            check_and_save(output_dir, filename, x_val, y_val)

    def __getitem__(self) :
        return

def check_and_save(out_dir, filename, x, y):
    if not os.path.isfile(filename):
        save_npzfile(out_dir, filename, x, y)
    else:
        print(f"{filename} is already exist!")

def save_npzfile(out_dir, filename, x, y) :
    create_folder(out_dir)
    save_dict = {
            "x" : x,
            "y" : y,
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
    parser.add_argument("--test_size", type=float, default=0.5); 
    parser.add_argument("--lower_freq", type=float, default=0.5)
    parser.add_argument("--high_freq", type=float, default=50); 
    parser.add_argument("--score_list", type=list, default=[1,6]); #[0,4] # [3] -> 0,1,2,3 /4,5,6,7,8,9     # [3,7] -> 0,1,2,3 / 4,5,6 / 7,8,9    # [1,6] -> 0,1 / 2,3,4,5 / 6,7,8,9
    parser.add_argument("--data_path", type=str, default='/opt/workspace/MS_DATA/PREPROCESSED_DATA/')
    parser.add_argument("--channel_num", type=int, default=28)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--one_bundle", type=int, default=int(1500/2)) # 500hz -> 3초에 1500행
    parser.add_argument("--output_path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_scale_raw_original')
    args = parser.parse_args()
    Load_Data(args)

if __name__ == "__main__" :
    main()