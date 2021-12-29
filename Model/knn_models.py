from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import data_loader
from get_args import Args

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

args_class = Args()
args = args_class.args

data = data_loader.Dataset(args, phase="train")
valid_data = data_loader.Dataset(args, phase="valid")

learner = ActiveLearner(
        # estimator=RandomForestClassifier(),
        estimator=knn,
        query_strategy=uncertainty_sampling,
        X_training=data.x, y_training=data.y
    )
query_index, query_instance = learner.query(valid_data.x, n_instances=10)
print(query_index)
learner.teach(X=valid_data.x[query_index], y=valid_data.y[query_index], only_new=True)

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(X=data.x)
x_component, y_component = transformed_data[:,0], transformed_data[:,1]
plt.figure(figsize=(8.5,6), dpi=130)
plt.scatter(x=x_component, y=y_component, c=data.y, cmap='viridis', s=50, alpha=8/10)
plt.savefig('d.png')