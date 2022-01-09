from numpy.core.fromnumeric import repeat
from sklearn.manifold import TSNE
import data_loader_5fold
from get_args import Args
import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

args_class = Args()
args = args_class.args

from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig("./ddd.png")
    # add the labels for each digit corresponding to the label
    txts = []

    # for i in range(num_classes):
    #     # Position of each label at median of data points.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)
    
    return f, ax, sc, txts

time_start = time.time()
data1 = np.load(f"/opt/workspace/xohyun/MS/Files/Single/Class3/Expt1/day1/subj01.npz")
x_subset = data1['x']
print(x_subset.shape)
x_subset = x_subset.reshape(-1,28)
y_subset = data1['y']
y_subset = y_subset.reshape(-1,1)
print(x_subset.shape)
print(y_subset.shape)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_subset)

print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

pca_df = pd.DataFrame(columns = ['pca1','pca2'])

pca_df['pca1'] = pca_result[:,0]
pca_df['pca2'] = pca_result[:,1]

top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component
fashion_scatter(top_two_comp.values,y_subset) # Visualizing the PCA output
