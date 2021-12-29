import numpy as np
import pandas as pd
import torch
import mne
import os
import filing
from setting import *

class Load_Data() :
    def __init__(self, data_path, subj_num) :
        data_list = os.listdir(data_path)
        data_list = sorted(data_list) # 순서대로 
        #print(data_list)
        ch_num = channel_num + 1
        original_data_list = [] 
        o_data_test = self.o_datalist = np.ones((1,1)) # 그냥 선언
       
        for _ in subj_num :
            print(_) # 몇 번째 파일인지 
    
            o_data = np.array(pd.read_csv(data_path + data_list[_], sep=',', header=0))
            #print(data_list[_])
            o_data_f = pd.DataFrame(o_data)
            filing.naming(o_data_f) 
            o_data_f.drop(['Time', 'ECG.(uV)', 'Resp', 'PPG', 'GSR', 'Packet Counter(DIGITAL)'], axis =  1, inplace = True) # 지우는 열 -> time도 포함시켜야 하는지 고민. 우선 data_loader.py에서 Time이용해서 조절하기 때문에 냅둠.
            # downsampling -> 250
            o_data_f = o_data_f.assign(g = o_data_f.index % 2).query('g==0') # 0으로 할건지 1로 할 건지인데, 0으로 해야 시작시간이 바르기 때무네..
            o_data_f.drop(['g'], axis = 1, inplace = True) # g열 버리고오
            o_data_f.reset_index(drop = True, inplace = True) # 인덱스 재설정 0부터~~

            #---# set class #---#
            # print("o_data_f\n", o_data_f['TRIGGER(DIGITAL)'])
            if class_num == 2:
                o_data_f['TRIGGER(DIGITAL)'] = o_data_f['TRIGGER(DIGITAL)'].apply(lambda x:0 if x <= score_list[0] else 1)
            elif class_num == 3:
                o_data_f['TRIGGER(DIGITAL)'] = o_data_f['TRIGGER(DIGITAL)'].apply(lambda x:0 if x <= score_list[0] else (2 if x >= score_list[1] else 1))
            # print("o_data_f_change\n", o_data_f['TRIGGER(DIGITAL)'])
            # print(o_data_f['TRIGGER(DIGITAL)'])
            o_data = o_data_f.to_numpy()                    # 다시 numpy 배열로
            cut = len(o_data) % (one_bundle * ch_num)  # 나머지 부분 잘라내기
        
            o_data = o_data[:-int(cut)].reshape(-1, one_bundle, ch_num) # [30, 7500, 30]
            #print("-o_data.shape-", o_data.shape)
            original_data_list.append(o_data)
  
        self.o_datalist = np.vstack(original_data_list) # 세로로 결합 train -> [1342, 7500, 30] / [682, 750, 30]
        #print("-o_datalist.shape-", self.o_datalist.shape)
        self.o_data_test = o_data_test # validation
        b, t, c = self.o_datalist.shape
        self.o_datalist = mne.filter.filter_data(self.o_datalist.reshape(-1, ch_num).T, sfreq = 250, l_freq = lower_freq, h_freq = high_freq, picks = np.s_[1:-2]).T.reshape(b, t, c)
        print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★", self.o_datalist.shape) # [464, 750, 29]
    def __getitem__(self) :
        return self.o_datalist