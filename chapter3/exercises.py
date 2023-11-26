from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from scipy.ndimage import shift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# # data
# mnist = fetch_openml("mnist_784", as_frame=False)
# X, y = mnist.data, mnist.target

# # split into train and test sets
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]



# 1.

# # get k nearest neighbours classifier
# knn_clf = KNeighborsClassifier()

# # parameter grid
# parameter_grid_knn = {
#     "weights": ["uniform", "distance"],
#     "n_neighbors": [2, 4, 6, 8, 10, 12, 14]
#     }

# # perform grid search
# grid_search = GridSearchCV(knn_clf,
#                            parameter_grid_knn,
#                            cv=3,
#                            scoring="accuracy")

# # fit to data and get best parameteres
# grid_search.fit(X_train, y_train)

# # make predictions
# pred = grid_search.predict(X_test)

# # get score
# print(f"Accuracy: {(pred == y_test).mean()}")  # got 97.14% accuracy





# 2.

# def shift_digit(digit, instruction=""):
#     reshaped_digit = digit.reshape(28,28)
#     vert_shift = float(instruction=="down") - float(instruction=="up")
#     horz_shift = float(instruction=="right") - float(instruction=="left")
#     return shift(reshaped_digit, [vert_shift, horz_shift], cval=0).reshape(784)

# def shift_data(data, instruction=""):
#     for i in range(data.shape[0]):
#         data[i] = shift_digit(data[i], instruction)
#     return data

# X_train_up = shift_data(X_train, "up")
# X_train_down = shift_data(X_train, "down")
# X_train_left = shift_data(X_train, "left")
# X_train_right = shift_data(X_train, "right")

# X_list = [X_train, X_train_up, X_train_down, X_train_left, X_train_right]

# # get expanded dataset and targets
# X_train_expanded = np.zeros(shape=(5*len(X_train), 784))
# y_train_expanded = np.array([str(w) for w in np.zeros(shape=(5*len(y_train)))])
# for i in range(5):
#     X_train_expanded[i*len(X_train):(i+1)*len(X_train)] = X_list[i]
#     y_train_expanded[i*len(X_train):(i+1)*len(X_train)] = y_train

# # shuffle indices to randomize
# shuffle_idx = np.random.permutation(len(X_train_expanded))
# #X_train_expanded_shuffled, y_train_expanded_shuffled = shuffle(X_train_expanded, y_train_expanded, random_state=42)  # leads to 0.9691
# X_train_expanded_shuffled = X_train_expanded[shuffle_idx]  # also results in 0.9691 using shuffle
# y_train_expanded_shuffled = y_train_expanded[shuffle_idx]

# # get best parameteres from previous question
# best_knn_clf = KNeighborsClassifier(**grid_search.best_params_)

# # train on bigger data
# best_knn_clf.fit(X_train_expanded_shuffled, y_train_expanded_shuffled)

# # make predictions
# pred = best_knn_clf.predict(X_test)

# # get accuracy
# print(f"Accuracy: {(pred == y_test).mean()}")
# print(f"KNN score: {best_knn_clf.score(X_test, y_test)}")




# 3.

# get data
titanic_data_train = pd.read_csv("datasets/titanic/train.csv")
titanic_data_test = pd.read_csv("datasets/titanic/test.csv")

# get survived column
survived_train = titanic_data_train["Survived"]
titanic_data_train = titanic_data_train.drop(["Survived"], axis=1)

print(titanic_data_test)
print(titanic_data_train)


# Preprocessing work

# get info about dataframe
titanic_data_train.info()

# get statistical information
print(titanic_data_train.describe())

# check for columns with NAs
print(titanic_data_train.isna().mean())  # cabin is 77% NA
print(titanic_data_train["Cabin"])
titanic_data_train = titanic_data_train.drop(["Cabin"], axis=1)  # dropped cabin since too many missing values
titanic_data_test = titanic_data_test.drop(["Cabin"], axis=1)

# get numerical and categorical columns
num_columns = list(titanic_data_train.select_dtypes(include="number").columns)
cat_columns = list(titanic_data_train.drop(num_columns, axis=1).columns)

# numerica data

# get correlation matrix
train_data_num_copy = titanic_data_train[num_columns]
train_data_num_copy["survived"] = survived_train
corr_matrix = train_data_num_copy.corr()
print(corr_matrix["survived"].sort_values(ascending=False))
high_corr_num_df = train_data_num_copy.loc[:,corr_matrix["survived"].abs() > 0.07]
high_corr_num_df = high_corr_num_df.drop("survived", axis=1)
num_columns = high_corr_num_df.columns


# categorical data

# check for important categorical columns
print(titanic_data_train[cat_columns])
cat_columns = titanic_data_train[cat_columns].drop(["Name", "Ticket"], axis=1).columns  # drop name which is not likely to be useful
print(cat_columns)
print(titanic_data_train[cat_columns].head(10))

# apply one hot encoding
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(titanic_data_train[cat_columns].values)  # had to reshape again, result is a scipy sparse matrix

# result is a sparse matrix
print(pd.DataFrame(housing_cat_1hot.toarray()))





# modify training and testing data
processed_col_names = [x for x in titanic_data_train.columns if x in num_columns or x in cat_columns]
titanic_data_train = titanic_data_train[processed_col_names]
titanic_data_test = titanic_data_test[processed_col_names]



# make pipeline

# make numerical pipeline
num_pipeline = Pipeline([
    ("num_imputer", SimpleImputer(strategy="median")),
    ("num_standardize", StandardScaler())
])

# make categorical pipeline
cat_pipeline = Pipeline([
    ("cat_imputer", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", OneHotEncoder(handle_unknown="ignore"))
])

# preprocess both numerical and categorical by using column transformer
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_columns),
    ("cat", cat_pipeline, cat_columns)
])

# run pipeline
titanic_data_train_prepared = preprocessing.fit_transform(titanic_data_train)
print(titanic_data_train_prepared)



# train model


# first we make pipeline for preprocessing followed by sgd
sgd_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("svc", SVC(random_state=42))
])


# make parameter grid
parameter_grid_svc = {
    "svc__kernel": ["linear", "rbf", "sigmoid"],
    "svc__gamma": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
    }

# get best parameters
grid_search = GridSearchCV(sgd_pipeline,
                           parameter_grid_svc,
                           cv=10,
                           scoring="accuracy")


# fit to data and get best parameteres
grid_search.fit(titanic_data_train, survived_train)

# get scores 
scores = cross_val_score(grid_search, titanic_data_train, survived_train, cv=3)

# make predictions
print(f"Best parameters: {grid_search.best_params_}")
print(f"Average cv-score: {scores.mean()}")  # got 0.8170594837261503











