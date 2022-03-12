import mat73
import argparse
import pandas as pd
import numpy as np
import mne 
from utils import *
from sklearn.model_selection import train_test_split

class Load_Data():
    def __init__(self, args):
        for video_num in range(1,21):
            for subj_num in range(1,16):
                try:
                    mat_file = mat73.loadmat(f'/opt/workspace/MS_DATA_public/VRSA-Shaking/Physiological signal/EEG/{video_num}/subj_{subj_num}.mat')
                    ssq = pd.read_csv('/opt/workspace/MS_DATA_public/VRSA-Shaking/Shaking_SSQ.csv')
                except:
                    print(f"======={video_num}, {subj_num}")
                    continue
                eeg = mat_file['final_array']
                eeg = eeg[2560:17920,] # 120s
                ssq = ssq.iloc[subj_num-1, video_num-1]

                #---# changing scale and to_numpy #---#
                o_data_f = pd.DataFrame(eeg)
                o_data_f.iloc[:,:] = o_data_f.iloc[:,:].apply(lambda x : x / 100)

                o_data = o_data_f.to_numpy()            # 다시 numpy 배열로
                o_data = o_data.reshape(-1, args.one_bundle, args.channel_num)
                
                #---# bandpass filtering #---#
                b, t, c = o_data.shape
                o_data = mne.filter.filter_data(o_data.reshape(-1, args.channel_num).T, sfreq = 250, l_freq = args.lower_freq, h_freq = args.high_freq, picks = np.s_[1:-2]).T.reshape(b, t, c)
                
                x = o_data
                if ssq < 19 : y = 0 
                elif ssq > 68 : y = 2
                else : y = 1
                                
                #--------------------------#
                #---# save single data #---#  
                #--------------------------#
                # output_dir = str(args.output_path) + "/Single/Class" + str(args.class_num) + "/Expt" + str(expt) + "/day" + str(day)
                # filename = "subj" + str(subj_num).zfill(2)
                # file_path_name = output_dir + "/" + filename + ".npz"
                # check_and_save(output_dir, filename, x, y)
                
                # #-------------------------#
                # #---# save split data #---#
                # #-------------------------#
                # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.test_size, shuffle=True, random_state=1004)
                # output_dir = str(args.output_path) + "/Split" + str(args.test_size) + "/Class" + str(args.class_num) + "/Expt" + str(expt) + "/day" + str(day)
                
                # filename = "subj" + str(subj_num).zfill(2) + "_train"
                # check_and_save(output_dir, filename, x_train, y_train)
                
                # filename = "subj" + str(subj_num).zfill(2) + "_val"
                # check_and_save(output_dir, filename, x_val, y_val)

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

def main() :
    parser = argparse.ArgumentParser(); 
    parser.add_argument("--test_size", type=float, default=0.1); 
    parser.add_argument("--lower_freq", type=float, default=0.5)
    parser.add_argument("--high_freq", type=float, default=50); 
    parser.add_argument("--score_list", type=list, default=[1,6]); #[0,4] # [3] -> 0,1,2,3 /4,5,6,7,8,9     # [3,7] -> 0,1,2,3 / 4,5,6 / 7,8,9    # [1,6] -> 0,1 / 2,3,4,5 / 6,7,8,9
    parser.add_argument("--data_path", type=str, default='/opt/workspace/MS_DATA/PREPROCESSED_DATA/')
    parser.add_argument("--channel_num", type=int, default=14)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--one_bundle", type=int, default=int(128*3)) # 128Hz * 3초씩 
    parser.add_argument("--output_path", type=str, default='/opt/workspace/xohyun/MS_codes/Files_scale_CS')
    args = parser.parse_args()
    Load_Data(args)

if __name__ == "__main__" :
    main()