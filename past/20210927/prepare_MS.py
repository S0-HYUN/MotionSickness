import argparse
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from setting import *
import mne
from sklearn.model_selection import train_test_split

def main() :
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type = str, default="/opt/workspace/soxo/Motionsickness_Data/PREPROCESSED_DATA/"")
    parser.add_argument("--test_size", type=float, default=0.5)
    args = parser.parse_args()
    test_size = args.test_size

    data_path = '/opt/workspace/soxo/Motionsickness_Data/PREPROCESSED_DATA/'
    datalist = os.listdir(data_path)
    datalist = sorted(datalist)

    ch_num = channel_num + 1

    for i in range(len(datalist)) : 
        if i % 2 == 0 : expt = 1
        else : expt = 2 

        if (i % 4 == 0) or (i % 4 == 1) : day = 1
        else : day = 2
        subj = (i // 4) + 1
        
        print(".....["+datalist[i]+"] loading.....")
        o_data = np.array(pd.read_csv(data_path + datalist[i], sep=',', header=0))
        o_data_f = pd.DataFrame(o_data)
        filing.naming(o_data_f) 
        o_data_f.drop(['Time', 'ECG.(uV)', 'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)'], axis =  1, inplace = True) # 지우는 열 -> time도 포함시켜야 하는지 고민. 우선 data_loader.py에서 Time이용해서 조절하기 때문에 냅둠.
        
        #---# downsampling -> 250 #---#
        o_data_f = o_data_f.assign(g = o_data_f.index % 2).query('g==0')    # 0으로 할건지 1로 할 건지인데, 0으로 해야 시작시간이 바르기 때무네..
        o_data_f.drop(['g'], axis = 1, inplace = True)                      # g열 버림
        o_data_f.reset_index(drop = True, inplace = True)                   # 인덱스 재설정 0부터~~

        #---# set class #---#
        if class_num == 2:
            o_data_f['TRIGGER(DIGITAL)'] = o_data_f['TRIGGER(DIGITAL)'].apply(lambda x:0 if x <= score_list[0] else 1)
        elif class_num == 3:
            o_data_f['TRIGGER(DIGITAL)'] = o_data_f['TRIGGER(DIGITAL)'].apply(lambda x:0 if x <= score_list[0] else (2 if x >= score_list[1] else 1))

        o_data = o_data_f.to_numpy()            # 다시 numpy 배열로
        cut = len(o_data) % one_bundle          # 나머지 부분 잘라내기 cut = len(o_data) % (one_bundle * ch_num) 여기 꼭 확인

        o_data = o_data[:-int(cut)].reshape(-1, one_bundle, ch_num)

        #---# bandpass filtering #---#
        b, t, c = o_data.shape
        o_data = mne.filter.filter_data(o_data.reshape(-1, ch_num).T, sfreq = 250, l_freq = lower_freq, h_freq = high_freq, picks = np.s_[1:-2]).T.reshape(b, t, c)
        
        x = o_data[:,:,:-1];    y = o_data[:,:,-1]
        #--------------------------#
        #---# save single data #---#
        #--------------------------#
        output_dir = "/opt/workspace/soxo/output/Single/Class" + str(class_num) + "/Expt" + str(expt)
        filename = "subj" + str(subj).zfill(2) + "_day" + str(day)
        save_npzfile(output_dir, filename, x, y)

        #------------------------#
        #---# save split data#---#
        #------------------------#
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=1004)
        
        output_dir = "/opt/workspace/soxo/output/Split" + str(test_size) + "/Class" + str(class_num) + "/Expt" + str(expt)
        filename = "subj" + str(subj).zfill(2) + "_day" + str(day) + "_train"
        save_npzfile(output_dir, filename, x_train, y_train)
        
        filename = "subj" + str(subj).zfill(2) + "_day" + str(day) + "_val"
        save_npzfile(output_dir, filename, x_val, y_val)

        raise

def createFolder(directory) :
    try: 
        if not os.path.exists(directory) :
            os.makedirs(directory)
    except OSError :
        print("Error: Creating directory" + directory)

def save_npzfile(out_dir, filename, x, y) :
    createFolder(out_dir)
    save_dict = {
            "x" : x,
            "y" : y,
    }
    np.savez(os.path.join(out_dir, filename), **save_dict)

if __name__ == "__main__" :
    main()

