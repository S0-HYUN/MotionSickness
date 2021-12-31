import torch
import numpy as np
# from tsnecuda import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

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
    scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=10, c=label, cmap='rainbow')
    plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(len(set(label)))])
    # plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(5,7)])
    # plt.show()
    plt.title(f"Experiment{expt}")
    plt.savefig(f"./plots/tsne_expt{expt}_all_features_newrange_1000.png")
    
    # if batch is not None:
    #     plt.savefig('visualization/{}.png'.format(batch), bbox_inches='tight')
    # plt.savefig('visualization/{}/{}_{}.png'.format(dataset, title, tde), bbox_inches='tight')