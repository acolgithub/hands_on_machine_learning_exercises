from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

import matplotlib.pyplot as plt
import numpy as np
import PIL



# k-means
X, y = make_blobs(n_samples=1000,
                  n_features=2,
                  centers=5,
                  cluster_std=[0.45, 0.65, 0.45, 0.45, 0.7],
                  random_state=42)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)


plt.figure(figsize=(6.5,5.5))
plt.scatter(X[:,0], X[:,1], s=0.5)
plt.xlabel(r"$x_{1}$", fontsize=12)
plt.ylabel(r"$x_{2}$", fontsize=12)
plt.grid()
plt.show()


# get predictions and compare with labels
print(y_pred)
print(y_pred is kmeans.labels_)

# get centroids found by algorithm
print(kmeans.cluster_centers_)


# assign new instances to nearest clouster
X_new = np.array([[0,2], [3,2], [-3,3], [-3,2.5]])
print(kmeans.predict(X_new), "\n")


# get distance from each instance to every centroid
print(kmeans.transform(X_new).round(2))



# Centroid initialization methods
good_init = np.array([[-3,3], [-3,2], [-3,1], [-1,2], [0,2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)

# get models inertia
print(kmeans.inertia_)

# get models score
print(kmeans.score(X))



# accelerated k-means and mini-batch k-means
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)

# get silhouette score
print(silhouette_score(X, kmeans.labels_), "\n")



# demonstration of opening file with PIL
# image = np.asarray(PIL.Image.open(filepath))
# print(image.shape) # returns array of three dimensions


# reshape array to long list of RGB colours
# X = image.reshape(-1,3)

# fit kmeans to data with 8 clusters
# kmeans = KMeans(n_clusters=8, random_state=42).fit(X)

# get segmented image containing nearest cluster center for each pixel
# segmented_img = kmeans.cluster_centers+[kmeans.labels_]

# reshape to original image shape
# segmented_img = segmented_img.reshape(image.shape)




# Using Clustering for Semi-Supervised Learning

X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_test, y_test = X_digits[1400:], y_digits[1400:]

# train logistic regression on 50 instances
n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])

# get accuracy on test set (labelled)
print(log_reg.score(X_test, y_test), "\n")

# cluster training set into 50 clusters
k = 50
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

# get distances to nearest cluster
X_digits_dist = kmeans.fit_transform(X_train)

# find image closest to centroid
representative_digit_idx = X_digits_dist.argmin(axis=0)
X_representative_digits = X_train[representative_digit_idx]

# manually label first 50 images
y_representative_digits = np.array([1, 3, 6, 0, 7, 9, 2, 4, 8, 9,
                                    5, 4, 7, 1, 2, 6, 1, 2, 5, 1,
                                    4, 1, 3, 3, 8, 8, 2, 5, 6, 9,
                                    1, 4, 0, 6, 8, 3, 4, 6, 7, 2,
                                    4, 1, 0, 7, 5, 1, 9, 9, 3, 7])

# check performance
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_representative_digits, y_representative_digits)
print(log_reg.score(X_test, y_test), "\n")

# label propagation
y_train_propagated = np.empty(len(X_train), dtype=np.int64)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]


# train model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train_propagated)
print(log_reg.score(X_test, y_test), "\n")



# eliminate outliers
percentile_closest = 99

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster &  above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

# train model and get accuracy
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg.score(X_test, y_test), "\n")


print((y_train_partially_propagated == y_train[partially_propagated]).mean(), "\n")





# DBSCAN
X, y = make_moons(n_samples=1000, noise=0.05)
dbscan =DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

# get all labels prediced from dbscan
print(dbscan.labels_, "\n")

# get indices of core samples
print(dbscan.core_sample_indices_, "\n")

# get core instances
print(dbscan.components_, "\n")

# train k neighbours classifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

# predict new cluster and get probabilities
X_new = np.array([[-0.5,0], [0,0.5], [1,-0.1], [2,1]])
print(knn.predict(X_new))
print(knn.predict_proba(X_new), "\n")

# get distances and indices of nearest neighbour
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1  # assign -1 if distance if large
print(y_pred.ravel(), "\n")




# Gaussian Mixtures
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

# get parameters
print(gm.weights_)
print(gm.means_)
print(gm.covariances_, "\n")

# check that algorithm converged and how many iterations it took
print(gm.converged_)
print(gm.n_iter_, "\n")

# use predict for hard clustering and predict_proba for soft clustering
print(gm.predict(X))
print(gm.predict_proba(X).round(3), "\n")



# sample new instance from model
X_new, y_new = gm.sample(6)
print(X_new)
print(y_new, "\n")


# estimate probability density function at point
print(gm.score_samples(X).round(2), "\n")



# Using Gaussian Mixtures for Anomaly Detection

densities = gm.score_samples(X)
density_threshold = np.percentile(densities,2)
anomalies = X[densities < density_threshold]

# compute bic and aic
print(gm.bic(X))
print(gm.aic(X), "\n")



# Bayesian Gaussian Mixture Models

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
print(bgm.weights_.round(2))























