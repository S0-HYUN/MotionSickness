from matplotlib.pyplot import axis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import fmin 
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from get_args_temp import Args
import data_loader.data_loader_active
import pandas as pd
from utils import * 
import mne

from sklearn import datasets
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
    df = pd.DataFrame(columns = ['test_subj', 'acc', 'f1', 'f1_weight'])
    for idx in range(1, 24) :
        print(idx)
        args.test_subj = idx

        data = data_loader.data_loader_active.Dataset(args, phase="train")
        x = data.x.permute(0,2,1) # [sample, 28, 750]
        # psds, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=0.2, fmax=3.99) #delta 

        psds_delta, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=0.2, fmax=3.99) #delta
        psds_delta = psds_delta.mean(axis=2) #[sample, 28]
        psds_theta, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=4, fmax=7.99) #theta
        psds_theta = psds_theta.mean(axis=2)
        psds_alpha, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=8, fmax=12.99) #alpha
        psds_alpha = psds_alpha.mean(axis=2)
        psds_beta, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=13, fmax=30) #beta
        psds_beta = psds_beta.mean(axis=2)
        psds_gamma, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=30) #gamma
        psds_gamma = psds_gamma.mean(axis=2)

        x_train = np.dstack([psds_delta, psds_theta, psds_alpha, psds_beta, psds_gamma])
        x_train = x_train.reshape(x_train.shape[0], -1)
        y_train = data.y.numpy()


        #### test data
        data_test = data_loader.data_loader_active.Dataset(args, phase="test")
        x = data_test.x.permute(0,2,1)

        psds_delta, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=0.2, fmax=3.99) #delta
        psds_delta = psds_delta.mean(axis=2) #[sample, 28]
        psds_theta, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=4, fmax=7.99) #theta
        psds_theta = psds_theta.mean(axis=2)
        psds_alpha, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=8, fmax=12.99) #alpha
        psds_alpha = psds_alpha.mean(axis=2)
        psds_beta, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=13, fmax=30) #beta
        psds_beta = psds_beta.mean(axis=2)
        psds_gamma, freqs = mne.time_frequency.psd_array_multitaper(x, sfreq=250, fmin=30) #gamma
        psds_gamma = psds_gamma.mean(axis=2)
   
        # tt = np.array(bands); tt = tt.reshape(channel_num, 5)
        # x_test = tt; y_test = data_test.y.numpy()
        x_test = np.dstack([psds_delta, psds_theta, psds_alpha, psds_beta, psds_gamma])
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_test = data_test.y.numpy()

        #---# RF #---#
        # forest = RandomForestClassifier(n_estimators=100)
        # forest.fit(x_train, y_train)

        #---# SVM #---#
        classifier = SVC(kernel='rbf', C = 1)
        classifier.fit(x_train, y_train)

        #---# KNN #---#
        # classifier = KNeighborsClassifier(n_neighbors = 3)
        # classifier.fit(x_train, y_train)

        # 예측
        # y_pred = forest.predict(x_test)
        y_pred = classifier.predict(x_test)

        # 정확도 확인
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        f1_weight = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        current_time = get_time()
        df.loc[idx-1] = [args.test_subj, acc, f1, f1_weight]
        create_folder(f"./csvs_traditional") # make folder
        df.to_csv(f'./csvs_traditional/{current_time}_MS_KNN_.csv', header = True, index = False)
        
        drawing_cm(args, cm)
        create_folder(f"./csvs/csvs_{args.model}_{args.standard}_{args.class_num}") # make folder
        df.to_csv(f'./csvs/csvs_{args.model}_{args.standard}_{args.class_num}/{current_time}_MS_{args.model}_subj{args.test_subj}_{args.standard}_{args.class_num}class_test.csv', header = True, index = False)
        print("-")


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