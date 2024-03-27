from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np


# get data
mnist = fetch_openml("mnist_784", as_frame=False)

# get train and test sets
X_train, y_train = mnist.data[:60000], mnist.target[:60000]
X_test, y_test = mnist.data[60000:], mnist.target[60000:]

# get pca
pca = PCA()

# fit to training data
pca.fit(X_train)

# get cumulative sum of explained variance ratio
cumsum = np.cumsum(pca.explained_variance_ratio_)

# find index to isolate pcas associated with 95% variance
d = np.argmax(cumsum >= 0.95) + 1  # d equals 154


# alternative
pca = PCA(n_components=0.95)  # find pcas assoicated with 95% variance
X_reduced = pca.fit_transform(X_train)


# number of pca components
print(pca.n_components_)



# make pipeline
clf = make_pipeline(
    PCA(random_state=42),
    RandomForestClassifier(random_state=42)
)

# get parameter distribution
param_distrib = {
    "pca__n_components": np.arange(10,80),
    "randomforestclassifier__n_estimators": np.arange(50,500)
}

# optimize parameters
rnd_search = RandomizedSearchCV(estimator=clf,
                                param_distributions=param_distrib,
                                n_iter=10,
                                cv=3,
                                random_state=42)

rnd_search.fit(X_train[:1000], y_train[:1000])

# get best paramaeters
print(rnd_search.best_params_)




# PCA for Compression

# decompress reduced MNIST
X_recovered = pca.inverse_transform(X_reduced)




# Randomized PCA

rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)

# transform data to reduced
X_reduced = rnd_pca.fit_transform(X_train)



# Incremental PCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)




# store in file on disk
filename = "my_mnist.mmap"
X_mmap = np.memmap(filename, dtype="float32", mode="write", shape=X_train.shape)
X_mmap[:] = X_train  # could be a look instead saving the data chunk by chunk
X_mmap.flush()


# load memmap file and use incremental PCA
X_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1,784)
batch_size = X_mmap.shape[0] // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mmap)




# Random Projection

m, eps = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=eps)
print(d)


n = 20000
np.random.seed(42)
P = np.random.randn(d, n)/np.sqrt(d)  # std dev = square root of variance


X = np.random.randn(m,n)  # generate a fake dataset
X_reduced = X @ P.T


gaussian_rnd_proj = GaussianRandomProjection(eps=eps, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)  # same result as above


# inverting
components_pinv = np.linalg.pinv(gaussian_rnd_proj.components_)
X_recovered = X_reduced @ pinv.T



# LLE
X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)














