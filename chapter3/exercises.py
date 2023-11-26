from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from scipy.ndimage import shift
import matplotlib.pyplot as plt
import numpy as np


# data
mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

# split into train and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]



# 1.

# get k nearest neighbours classifier
knn_clf = KNeighborsClassifier()

# parameter grid
parameter_grid_knn = {
    "weights": ["uniform", "distance"],
    "n_neighbors": [2, 4, 6, 8, 10, 12, 14]
    }

# perform grid search
grid_search = GridSearchCV(knn_clf,
                           parameter_grid_knn,
                           cv=3,
                           scoring="accuracy")

# fit to data and get best parameteres
grid_search.fit(X_train, y_train)

# make predictions
pred = grid_search.predict(X_test)

# get score
print(f"Accuracy: {(pred == y_test).mean()}")  # got 97.14% accuracy





# 2.

def shift_digit(digit, instruction=""):
    reshaped_digit = digit.reshape(28,28)
    vert_shift = float(instruction=="down") - float(instruction=="up")
    horz_shift = float(instruction=="right") - float(instruction=="left")
    return shift(reshaped_digit, [vert_shift, horz_shift], cval=0).reshape(784)

def shift_data(data, instruction=""):
    for i in range(data.shape[0]):
        data[i] = shift_digit(data[i], instruction)
    return data

X_train_up = shift_data(X_train, "up")
X_train_down = shift_data(X_train, "down")
X_train_left = shift_data(X_train, "left")
X_train_right = shift_data(X_train, "right")

X_list = [X_train, X_train_up, X_train_down, X_train_left, X_train_right]

# get expanded dataset and targets
X_train_expanded = np.zeros(shape=(5*len(X_train), 784))
y_train_expanded = np.array([str(w) for w in np.zeros(shape=(5*len(y_train)))])
for i in range(5):
    X_train_expanded[i*len(X_train):(i+1)*len(X_train)] = X_list[i]
    y_train_expanded[i*len(X_train):(i+1)*len(X_train)] = y_train

# shuffle indices to randomize
shuffle_idx = np.random.permutation(len(X_train_expanded))
#X_train_expanded_shuffled, y_train_expanded_shuffled = shuffle(X_train_expanded, y_train_expanded, random_state=42)  # leads to 0.9691
X_train_expanded_shuffled = X_train_expanded[shuffle_idx]  # also results in 0.9691 using shuffle
y_train_expanded_shuffled = y_train_expanded[shuffle_idx]

# get best parameteres from previous question
best_knn_clf = KNeighborsClassifier(**grid_search.best_params_)

# train on bigger data
best_knn_clf.fit(X_train_expanded_shuffled, y_train_expanded_shuffled)

# make predictions
pred = best_knn_clf.predict(X_test)

# get accuracy
print(f"Accuracy: {(pred == y_test).mean()}")
print(f"KNN score: {best_knn_clf.score(X_test, y_test)}")










