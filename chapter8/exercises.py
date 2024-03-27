


# 1.
print(f"""Reducing a dataset's dimensionality allows for algorithms to perform faster and more efficiently.
The main drawback would be the loss of information in try to reduce the dimensions.\n""")



# 2.
print(f"""The curse of dimensionality refers to the increase of the concentration of points near the boundary of a cube as the dimension increases.
That is, as dimension increases a point in a cube is more likely near the boundary.\n""")



# 3.
print(f"""Reducing dimension of data if not reversible since you are usually losing information by projecting down in some form.\n""")



# 4.
print(f"""If a dataset is very nonlinear you could try adding more features in order to linearize the data and then apply PCA.
This allows PCA to work for a nonlinear dataset.
However, this may not work in general.\n""")



# 5.
print(f"""The resulting dataset could have dimenion anywhere between 1 and 950 dpending on the explained variance of each PCA component.
Note that the extreme situations include one PCA component with explained variance of 95% with the rest explaining 5% and 950 each explaining 0.1% each and the rest explaining 0.1% each.
Note that PCAs are arranged in descending order.\n""")



# 6.
print(f"""We would use standard PCA when the training dataset is not too large.
We would use incremental PCA when the training dataset is too large to fit into memory or when we want to apply PCA online.
In such a case, we would still want the number of desired PCA components to be small in order for this to be efficient.
We would use randomized PCA when the number of features is large but the number of components we want is small.
We would use randomized projection when the number of features is very large.\n""")



# 7.
print(f"""We could look at the percentage of variance explained by the new features.
This will assign a numerical value to the amount of information preserved.""")



# 8.
print(f"""Doing this makes sense if we want to quickly decrease the size of the data and then provide a more accurate reduction of dimensions on the resulting data.""")



# 9.
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

import pandas as pd
import time

mnist = fetch_openml("mnist_784", as_frame=True)

# obtain feature and target sets
X, y = mnist.data, mnist.target

# split into training and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# get classifier
forest_clf = RandomForestClassifier(criterion="gini", random_state=42)

# start time
start_time = time.time()

# train classifier
forest_clf.fit(X_train, y_train)

# evaluate model
print(f"Test set accuracy: {forest_clf.score(X_test, y_test)}")
print(f"Time taken: {time.time()-start_time}s\n")



# get PCA
forest_pca = PCA(n_components=0.95, random_state=42)

# fit to data
X_train_transformed = forest_pca.fit_transform(X_train)
X_test_transformed = forest_pca.transform(X_test)

# get classifier
forest_clf = RandomForestClassifier(criterion="gini", random_state=42)

# start time
start_time = time.time()

# train classifier
forest_clf.fit(X_train_transformed, y_train)

# evaluate model
print(f"Test set accuracy: {forest_clf.score(X_test_transformed, y_test)}")
print(f"Time taken: {time.time()-start_time}s\n")
print(f"Training was not faster.")
print(f"New classifier is slightly less accurate.\n")




# get classifier
sgd_clf = SGDClassifier(loss="hinge", penalty="l2", alpha=0.01, random_state=42)

# start time
start_time = time.time()

# train classifier
sgd_clf.fit(X_train, y_train)

# evaluate model
print(f"Test set accuracy: {sgd_clf.score(X_test, y_test)}")
print(f"Time taken: {time.time()-start_time}s\n")



# get PCA
sgd_pca = PCA(n_components=0.95, random_state=42)

# fit to data
X_train_transformed = sgd_pca.fit_transform(X_train)
X_test_transformed = sgd_pca.transform(X_test)

# get classifier
sgd_clf = SGDClassifier(loss="hinge", penalty="l2", alpha=0.01, random_state=42)

# start time
start_time = time.time()

# train classifier
sgd_clf.fit(X_train_transformed, y_train)

# evaluate model
print(f"Test set accuracy: {sgd_clf.score(X_test_transformed, y_test)}")
print(f"Time taken: {time.time()-start_time}s\n")
print(f"Training was much faster.")
print(f"New classifier is slightly more accurate.")



# 10.
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import numpy as np

# TSNE
X_embedded  = TSNE(n_components=2, learning_rate="auto", random_state=42).fit_transform(X[:5000])

plt.figure(figsize=(12.5, 10.5))
plt.scatter(X_embedded[:,0],
            X_embedded[:,1],
            c=y[:5000].astype(np.int8),  # colour based on digit
            cmap="jet",  # use cjet colours
            alpha=0.5)
plt.colorbar()  # add colourbar
plt.title("TSNE Plot")
plt.show()


# PCA
pca_reduction = PCA(n_components=2, random_state=42)
X_pca_embedded = pca_reduction.fit_transform(X[:5000])

plt.figure(figsize=(12.5, 10.5))
plt.scatter(X_pca_embedded[:,0],
            X_pca_embedded[:,1],
            c=y[:5000].astype(np.int8),
            cmap="jet",
            alpha=0.5)
plt.colorbar()
plt.title("PCA Plot")
plt.show()


# LLE
lle = LocallyLinearEmbedding(n_components=2, random_state=42)

X_lle = lle.fit_transform(X[:5000])

plt.figure(figsize=(12.5, 10.5))
plt.scatter(X_lle[:,0],
            X_lle[:,1],
            c=y[:5000].astype(np.int8),
            cmap="jet",
            alpha=0.5)
plt.colorbar()
plt.title("LLE Plot")
plt.show()


# MDS
mds = MDS(n_components=2, random_state=42)

X_mds = mds.fit_transform(X[:5000])

plt.figure(figsize=(12.5, 10.5))
plt.scatter(X_mds[:,0],
            X_mds[:,1],
            c=y[:5000].astype(np.int8),
            cmap="jet",
            alpha=0.5)
plt.colorbar()
plt.title("MDS Plot")
plt.show()


























