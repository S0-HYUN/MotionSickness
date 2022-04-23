from matplotlib.pyplot import axis

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import fmin 

from get_args_temp import Args
import data_loader.data_loader_active
import pandas as pd
from utils import * 
import mne

from sklearn import datasets
from mne.decoding import CSP
'''
gamma : 30 ~
beta : 13-30
alpha : 8-12.99
theta : 4-7.99
delta : 0.2-3.99
'''

def main():
    args_class = Args()
    args = args_class.args
    df = pd.DataFrame(columns = ['test_subj', 'acc'])
    for idx in range(1, 24) :
        print(idx)
        args.test_subj = idx

        data = data_loader.data_loader_active.Dataset(args, phase="train")
        x = data.x.permute(0,2,1) # [sample, 28, 750]
        y_train = data.y.numpy()
        # psds, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=0.2, fmax=3.99) #delta 
        info = mne.create_info(ch_names = naming(), sfreq = 250, ch_types="eeg")
        
        x = mne.EpochsArray(x, info=info)
        epochs_data = x.get_data()
        csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        x_train = csp.fit_transform(epochs_data, y_train)
        lda.fit(x_train, y_train)
        
        ### test
        data_test = data_loader.data_loader_active.Dataset(args, phase="test")
        x = data_test.x.permute(0,2,1)
        y_test = data_test.y.numpy()
        x_test = mne.EpochsArray(x, info=info)
        epochs_data_test = x_test.get_data()
        # csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
        x_test = csp.transform(epochs_data_test)
        # k = lda.fit_transform(x_test)
        # print(k)
        acc = lda.score(x_test, y_test)
        
        # 정확도 확인
        # acc = metrics.accuracy_score(y_test, y_pred)
        # f1 = metrics.f1_score(y_test, y_pred, average='micro')
        # f1_weight = metrics.f1_score(y_test, y_pred, average='weighted')
        current_time = get_time()
        df.loc[idx-1] = [args.test_subj, acc]
        create_folder(f"./csvs_traditional") # make folder
        df.to_csv(f'./csvs_traditional/{current_time}_MS_KNN_.csv', header = True, index = False)
        print("-")

def naming():
    name = ['Fp1(uV)', 'Fp2(uV)', 'AF3(uV)','AF4(uV)', 'F7(uV)', 
        'F8(uV)', 'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
        'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)',	'P8(uV)', 'P3(uV)',	'Pz(uV)',
        'P4(uV)', 'PO7(uV)', 'PO8(uV)',	'PO3(uV)', 'PO4(uV)', 'O1(uV)',	'O2(uV)']
    return name

if __name__ == "__main__" :
    main()



'''iris = datasets.load_iris()
    data = pd.DataFrame(
        {
            'sepal length': iris.data[:, 0],
            'sepal width': iris.data[:, 1],
            'petal length': iris.data[:, 2],
            'petal width': iris.data[:, 3],
            'species': iris.target
        }
    )
    from sklearn.model_selection import train_test_split
 
    x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
    y = data['species'] 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)'''

# x_train = psds; y_train = data.y.numpy()
    
        # data_test = data_loader.data_loader_active.Dataset(args, phase="test")
        # # x = data_test.x.reshape(data_test.x.shape[0], -1)
        # x = data_test.x.permute(0,2,1)
        # psds, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250)
        # x_test = psds; y_test = data_test.y.numpy()

        # x_train = x_train.reshape(x_train.shape[0], -1)
        # x_test = x_test.reshape(x_test.shape[0], -1)

        
        
        


# for trial in range(x.shape[0]):
        #     for ch in range(channel_num):
        #         psds_delta, freqs = mne.time_frequency.psd_array_multitaper(x[trial, ch, :], sfreq=250, fmin=0.2, fmax=3.99) #delta
        #         bands.append(psds_delta.mean())
        #         psds_theta, freqs = mne.time_frequency.psd_array_multitaper(x[trial, ch, :], sfreq=250, fmin=4, fmax=7.99) #theta
        #         bands.append(psds_theta.mean())
        #         psds_alpha, freqs = mne.time_frequency.psd_array_multitaper(x[trial, ch, :], sfreq=250, fmin=8, fmax=12.99) #alpha
        #         bands.append(psds_alpha.mean())
        #         psds_beta, freqs = mne.time_frequency.psd_array_multitaper(x[trial, ch, :], sfreq=250, fmin=13, fmax=30) #beta
        #         bands.append(psds_beta.mean())
        #         psds_gamma, freqs = mne.time_frequency.psd_array_multitaper(x[trial, ch, :], sfreq=250, fmin=30) #gamma
        #         bands.append(psds_gamma.mean())
            
            # tt = np.array(bands) #; tt = tt.reshape(channel_num, 5)
            # channel_avg.append(tt) # 28x5 = 140이 하나씩 들어감

    # channel_avg = np.array(channel_avg)
    # x_train = channel_avg.mean(axis=0); y_train = data.y.numpy()