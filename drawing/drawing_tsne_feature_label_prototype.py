import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
# from tsnecuda import TSNE
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *
import pandas as pd

def main():
    # data = np.load(f"/opt/workspace/xohyun/MS_codes/features_EEGNet_10/original_subj01_epoch10.npz")
    data = np.load(f"/opt/workspace/xohyun/MS_codes/features_rest/original_subj01.npz")
    x_raw = data['arr_0']; y_raw = data['arr_1']
    y_raw = np.repeat(1, y_raw.shape[0])
    x_proto = [x_raw.mean(axis=0).tolist()]
    y_proto = np.repeat(1,1)
    x_proto_dict = {"1":x_proto}

    for i in range(2,24):
        # data = np.load(f"/opt/workspace/xohyun/MS_codes/features_EEGNet_10/original_subj{str(i).zfill(2)}_epoch10.npz")
        data = np.load(f"/opt/workspace/xohyun/MS_codes/features_rest/original_subj{str(i).zfill(2)}.npz")
        # x_raw = torch.cat((x_raw, data['arr_0']))
        x_raw = np.concatenate((x_raw, data['arr_0']))
        # y_raw = np.concatenate((y_raw, data['arr_1']))
        y_raw = np.concatenate((y_raw, np.repeat(i, data['arr_0'].shape[0])))
        x_proto = np.concatenate((x_proto, [data['arr_0'].mean(axis=0)]))
        y_proto = np.concatenate((y_proto, np.repeat(i,1)))
        x_proto_dict[f'{str(i)}'] = [data['arr_0'].mean(axis=0).tolist()]

    #---# Save the prototype in json format #---#
    import json
    with open("./rest_prototype.json", "w") as f:
        json.dump(x_proto_dict, f)
    
    #---# choose label #---#
    # x_df = pd.DataFrame(data_x)
    # y_df = pd.DataFrame(data_y, columns=['label'])

    # choose_idx = y_df['label'].isin([5.0,6.0])
    # y_df = y_df.loc[choose_idx]
    # x_df = x_df.loc[choose_idx]
    # # print(x_df)
    # # print("\n", y_df)
    # data_x = x_df.to_numpy()
    # data_y = y_df.to_numpy()

    # visualizer(x_raw, y_raw)
    visualizer(x_proto, y_proto)


def visualizer(data, label, batch=None, dataset=None, tde=-1, title=None):
    expt = 1

    label = np.array(label, dtype=np.int32)
    # data, label = data[7000:8000,:], label[7000:8000]
    print('T-SNE starts!')
    # data, label = data.cpu().data.numpy(), label
    x_embedded = TSNE(n_components=2, perplexity=30, learning_rate=10, n_iter=1000).fit_transform(data)
    plt.figure(figsize=(12, 8))
    print('T-SNE finished!')
    # label = list(label.squeeze(1))
    scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=25, c=label, cmap='rainbow')
    plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(len(set(label)))])
    # plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(5,7)])
    # plt.show()

    current_time = get_time()
    plt.title(f"Experiment{expt}")
    plt.savefig(f"./plots/{current_time}_tsne_expt{expt}_all_features.png")

if __name__ == "__main__" :
    main()