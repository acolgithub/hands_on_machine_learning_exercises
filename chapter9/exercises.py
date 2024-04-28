



# Q1: How would you define clustering? can you name a few clustering
#     algorithms?

print(f"""Clustering is finding similar groups among unlabelled data.
Some clustering algorithms include K-means and DBSCAN.\n""")



# Q2: What are some of the main applications of clustering algorithms?

print(f"""Some of the main application of clustering algorithms include
customer segmentation, data analysis, and dimensionality reduction.\n""")



# Q3: Describe two techniques to select the right number of clusters when
#     using k-means.

print(f"""One technique to select the right number of clusters is to plot
inertia as a function of the number of clusters and choose the number of
clusters which maximizes decline.
That is, after this value the declien in inertia should be smaller.
Note that inertia calculates the sum of squared distances between
the instances and their closest centroids.
A better approach is to plot the silhouette score as a function of the
number of clusters.
We would choose the value that maximizes the silhouette score (or is close
to maximum).
Note that silhouette coefficients account for mean intra-cluster distance
and mean nearest-cluster distance.\n""")



# Q4: What is label propagation? Why would you implement it, and how?

print(f"""Label propagation is the extension of the labels given to a
subset of the data to the rest of the instances in the same cluster.
That is, we extend a label of some data point to the other members of
the same cluster.
We would implement this in a situation where it would be difficult to
manually label all data instances.\n""")



# Q5: Can you name two clustering algorithms that can scale to large datasets?
#     And two that look for regions of high density?

print(f"""Two clustering algorithms that scale to large datasets are
BIRCH and mini-batch k-means.
Two algorithms which look for regions of high density include DBSCAN and
agglomerative clustering.\n""")



# Q6: Can you think of a use case where active learning would be useful? How
#     would you implement it?

print(f"""Active learning would be useful when trying to classify
some sort of object.
If the program becomes uncertain then a human expert can provide
a label.
You could have the learning algorithm produce a specific syntax
to indicate that it needs input from an expert.
After providing the answer the program can then use the result as
a training instance.\n""")



# Q7: What is the difference between anomaly detection and novelty
#     detection?

print(f"""In anomaly detection we are trying to detect when a particular
data point represents an outlier.
In novelty detection we are interested in finding an unusual
occurrence.
It differs from anomaly detection in that it assumes it is trained
on a 'clean' dataset without outliers.\n""")



# Q8: What is a Gaussian mixture? What tasks can you use it for?

print(f"""A Gaussian mixture assumes that the instances were generated
from a mixture of several Gaussian distributions whose parameters
are unknown.
You can use this for anomaly detection.\n""")



# Q9: Can you name two techniques to find the right number of clusters when
#     using a Gaussian mixture model?

print(f"""Two techniques to find the right number of clusters when
using a Gaussian mixture model include Bayesian information
criterion (BIC) or Akaike information criterion (AIC).\n""")



# Q10: The classic Olivetti faces dataset contains 400 grayscale 64 x 64-pixel
#      images of faces. Each image is flattened to a 1D vector of size 4,096.
#      Forty different people were photographed (10 times each), and the usual
#      task is to train a model that can predict which person is represented in
#      each picture. Load the dataset using the
#      sklearn.datasets.fetch_olivetti_faces() function, then split it into a
#      training set, a validation set, and a test set (note that the dataset is
#      already scaled between 0 and 1). Since the dataset is quite small, you
#      will probably want to use stratified sampling to ensure that there are the
#      same number of images per person in each set. Next cluster the images
#      using k-means, and ensure that you have a good number of clusters
#      (using one of the techniques discussed in this chapter). Visualize the
#      clusters: do you see similar faces in each cluster?

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import numpy as np

# data
data = fetch_olivetti_faces()

# data and target
X, y = data.data, data.target

# get training stratified shuffle split
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=42)

# get train/int set
X_train, X_int = [], []
y_train, y_int = [], []
for i, (train_index, int_index)  in enumerate(strat_split.split(X, y)):
    X_train, X_int = X[train_index], X[int_index]
    y_train, y_int = y[train_index], y[int_index]

# get validation/test stratified shuffle split
strat_split2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)

# get val/test set
X_val, X_test = [], []
y_val, y_test = [], []
for i, (val_index, test_index) in enumerate(strat_split2.split(X_int, y_int)):
    X_val, X_test = X_int[val_index], X_int[test_index]
    y_val, y_test = y_int[val_index], y_int[test_index]

ks = range(100,110)
inertia = []
silhouette_scores = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_train, kmeans.labels_))


plt.figure(figsize=(12.5,10.5))
plt.plot(ks, inertia, c="r")
plt.xlabel("k")
plt.ylabel("inertia")
plt.title("Inertia plot")
plt.show()

plt.figure(figsize=(12.5,10.5))
plt.plot(ks, silhouette_scores, c="b")
plt.xlabel("k")
plt.ylabel("silhouette score")
plt.title("Silhouette score plot")
plt.show()

# get kmeans
kmeans = KMeans(n_clusters=104, random_state=42)
kmeans.fit(X_train) 

# plot faces function
def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)  # reshape to be 64 by 64
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))  # set figure size based on number of rows and columns
    for index, (face, label) in enumerate(zip(faces, labels)):  # zip together faces and labels
        plt.subplot(n_rows, n_cols, index + 1)  # make subplots of given number of rows and columns
        plt.imshow(face, cmap="gray")  # show face in gray scale
        plt.axis("off")  # remove axis
        plt.title(label)  # add title
    plt.show()

for cluster_id in np.unique(kmeans.labels_):  # loop over cluster ids
    print("Cluster", cluster_id)  # print cluster number
    in_cluster = kmeans.labels_==cluster_id  # find labels matching cluster id
    faces = X_train[in_cluster]  # get faces corresponding to cluster
    labels = y_train[in_cluster]  # get labels corresponding to cluster
    plot_faces(faces, labels)  # plot face


# Q11: Continuing with the Olivetti faces dataset, train a classifier to predict
#      which person is represented in each picture, and evaluate it on the
#      validation set. Next, use k-means as a dimensionality reduction tool, and
#      train a classifier on the reduced set. Search for the number of clusters
#      that allows the classifier to get the best performance: what performance
#      can you reach? What if you append the features from the reduced set to
#      the original features (again, searching for the best number of clusters)?

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# get random forest classifier
forest_clf = RandomForestClassifier(criterion="gini", random_state=42)

# train classifier
forest_clf.fit(X_train, y_train)

# make predictions
y_pred = forest_clf.predict(X_val)
print(f"Score on validation set: {accuracy_score(y_val, y_pred)}")  # got 92.5% accuracy


# transform data to reduced dimensional model
X_train_reduced = kmeans.transform(X_train)
X_val_reduced = kmeans.transform(X_val)
X_test_reduced = kmeans.transform(X_test)

# get random forest classifier
forest_clf = RandomForestClassifier(criterion="gini", random_state=42)

# train classifier
forest_clf.fit(X_train_reduced, y_train)

# make predictions
y_pred = forest_clf.predict(X_val_reduced)
print(f"Score on validation set: {accuracy_score(y_val, y_pred)}")  # got 72.5% accuracy




# make pipeline
clf = make_pipeline(
    KMeans(random_state=42),
    RandomForestClassifier(criterion="gini", random_state=42)
)

# parameter grid
parameter_grid = {
    "kmeans__n_clusters": np.arange(2,160)
}

# find optimal number of clusters
grid_search = GridSearchCV(estimator=clf,
                           param_grid=parameter_grid,
                           cv=3)

grid_search.fit(X_train, y_train)

# get best model
opt_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

y_pred = opt_model.predict(X_val)
print(f"Validation accuracy: {accuracy_score(y_val, y_pred)}")  # got 85% accuracy







# append features from reduced set to original features
X_train_concat = np.c_[X_train, X_train_reduced]
X_val_concat = np.c_[X_val, X_val_reduced]
X_test_concat = np.c_[X_test, X_test_reduced]

# make pipeline
clf = RandomForestClassifier(criterion="gini", random_state=42)

# fit model
clf.fit(X_train_concat, y_train)

# make predictions
y_pred = clf.predict(X_val_concat)
print(f"Validation accuracy: {accuracy_score(y_val, y_pred)}")  # got 91.25%




# Q12: Train a Gaussian mixture model on the Olivetti faces dataset. To speed
#      up the algorithm, you should probably reduce the datsset's
#      dimensionality (e.g., use PCA, preserving 99% of the variance). Use the
#      model to generate some new faces (using the sample() method), and
#      visualize them (if you used PCA, you will need to use its
#      inverse_transform() method). Try to modify some images (e.g., rotate,
#      flip, darken) and see if the model can detect the anomalies (i.e., compare
#      the output of the score_samples() method for normal images and for
#      anomalies).

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# get pca
pca = PCA(n_components=0.99, random_state=42)

# transform data
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# get Gaussian mixture model
gmm = GaussianMixture(n_components=40, random_state=42)

# fit model
gmm.fit(X_train_pca, y_train)

# generate new faces
X_new, y_new = gmm.sample(10)

# invert PCA
X_new_no_pca = pca.inverse_transform(X_new)
plot_faces(X_new_no_pca, y_new)

















