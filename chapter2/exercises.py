from tar_file_opener import load_tgz_data

# class
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import MetaEstimatorMixin
from sklearn.utils.validation import check_array


# data
housing_data = load_tgz_data(tarfile_input="housing.tgz")
print(housing_data)

# use cut function to decompose median income into suitable intervals
housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                               bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])

# create train, test strata
strat_train_set, strat_test_set = train_test_split(housing_data,
                                                   test_size=0.2,
                                                   stratify=housing_data["income_cat"],
                                                   random_state=42)

# form housing labels and drop from data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# drop categorical income column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# cluster similarity class
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters  # set number of clusters
        self.gamma = gamma  # set gamma
        self.random_state = random_state  # set random state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# pipeline
# get column ratio
def column_ratio(X):
    return X[:,[0]]/X[:,[1]]

# get ratio name
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

# define ratio pipeline
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),  # first impute with median
        FunctionTransformer(column_ratio, feature_names_out=ratio_name), # transform into ratio
        StandardScaler()  # apply scaler
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),  # impute values
    FunctionTransformer(np.log, feature_names_out="one-to-one"),  # apply logarithm
    StandardScaler()  # apply scaler
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),  # divide total bedrooms by total rooms
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),  # divide total rooms by number of households
    ("people_per_house", ratio_pipeline(), ["population", "households"]),  # divide population size by number of houses
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
    "households", "median_income"]),  # apply logarithm to fix large tails
    ("geo", cluster_simil, ["latitude", "longitude"]),  # apply clustering algorithm to see density
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),  # pass through categorical pipeline
],
remainder=default_num_pipeline) # one column remaining: housing_median_age


# # run pipeline
# housing_prepared = preprocessing.fit_transform(housing_data)
# print(housing_prepared.shape)

# print(preprocessing.get_feature_names_out())


# 1.
kernels = ["linear", "rbf"]
C = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]

housing_shrunk = housing_data[:5000]
housing_labels_shrunk = housing_labels[:5000]

def svm_test(kernels, C, gamma="scale"):
    y = list(np.zeros(len(C), dtype=int))
    for k in kernels:
        for c in C:
            svm_pip = make_pipeline(preprocessing, SVR(kernel=k, C=c, gamma=gamma))
            svm_rmses = -cross_val_score(svm_pip,
                                         housing_shrunk,
                                         housing_labels_shrunk,
                                         scoring="neg_root_mean_squared_error",
                                         cv=2)
            print(f"SVM test with {k} kernel, C={c} has score {0.5*svm_rmses}.")
            y[C.index(c)] = svm_rmses
        
        print(f"\nTest for {k} completed.\n")
        fig_name = f"figures/exercise1_{k}.png"
        fig = plt.figure(figsize=(6.5, 5.5))
        plt.plot(C, y)
        plt.grid()
        plt.xlabel("C parameter")
        plt.ylabel("RMSE score")
        plt.legend(labels=["RMSE 1", "RMSE 2", "RMSE 3"])
        plt.savefig(fig_name)
        plt.close()

# svm_test(kernels=kernels, C=C)

# paramter_grid = [
#     {"svr__kernel": ["linear"],
#      "svr__C": C},
#     {"svr__kernel": ["rbf"],
#      "svr__C": C}
# ]
# full_svm_pipeline = make_pipeline(preprocessing, SVR())
# grid_search = GridSearchCV(full_svm_pipeline, paramter_grid, cv=3,
#                            scoring="neg_root_mean_squared_error")
# grid_search.fit(housing_shrunk, housing_labels_shrunk)
# print(f"best parameters: {grid_search.best_params_}")
# print(f"best score: {grid_search.best_score_}")



# param_grid = [
#         {'svr__kernel': ['linear'], 'svr__C': C},
#         {'svr__kernel': ['rbf'], 'svr__C': C}
#     ]

# svr_pipeline = make_pipeline(preprocessing, SVR())
# grid_search = GridSearchCV(svr_pipeline, param_grid, cv=2,
#                            scoring='neg_root_mean_squared_error')
# grid_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

# svr_grid_search_rmse = -grid_search.best_score_
# print(svr_grid_search_rmse)

# print(grid_search.best_params_)


# 2.

# paramter_grid = [
#     {"svr__kernel": ["linear"],
#      "svr__C": C},
#     {"svr__kernel": ["rbf"],
#      "svr__C": C}
# ]
# full_svm_pipeline = make_pipeline(preprocessing, SVR())
# grid_search = RandomizedSearchCV(full_svm_pipeline, paramter_grid,
#                                  n_iter=50, cv=3,
#                                  scoring="neg_root_mean_squared_error")
# grid_search.fit(housing_shrunk, housing_labels_shrunk)
# print(f"best parameters: {grid_search.best_params_}")
# print(f"best score: {grid_search.best_score_}")



# 3.


# def svm_test_select(kernels, C, gamma="scale"):
#     y_select = list(np.zeros(len(C), dtype=int))
#     for k in kernels:
#         for c in C:
#             svm_pip_select = make_pipeline(preprocessing,
#                                            SelectFromModel(estimator=RandomForestRegressor(),
#                                                            threshold="mean",
#                                                            prefit=False),
#                                            SVR(kernel=k, C=c, gamma=gamma))
#             svm_rmses_select = -cross_val_score(svm_pip_select,
#                                          housing_shrunk,
#                                          housing_labels_shrunk,
#                                          scoring="neg_root_mean_squared_error",
#                                          cv=3)
#             print(f"SVM test with {k} kernel, C={c} has score {svm_rmses_select}.")
#             y_select[C.index(c)] = svm_rmses_select
        
#         print(f"\nTest for {k} completed.\n")
#         fig_name = f"figures/exercise3_{k}.png"
#         fig = plt.figure(figsize=(6.5, 5.5))
#         plt.plot(C, y_select)
#         plt.grid()
#         plt.xlabel("C parameter")
#         plt.ylabel("RMSE score")
#         plt.legend(labels=["RMSE 1", "RMSE 2", "RMSE 3"])
#         plt.savefig(fig_name)
#         plt.close()

# svm_test_select(kernels=kernels, C=C)



# 4.

# cluster similarity class

# neighbors similarity class attempt
class NearestNeighbourSimilarity(MetaEstimatorMixin, BaseEstimator, TransformerMixin):  # need metaestimatormixin for estimator constructor
    def __init__(self, estimator):
        self.estimator = estimator  # set estimator

    def fit(self, X, y=None):
        estimator_ = clone(self.estimator)  # cannot modify self.estimator
        estimator_.fit(X, y)  # fit to data, needed clone since cannot mutate parameter during fit
        self.estimator_ = estimator_  # copy fitted estimator to estimator_
        self.n_features_in_ = estimator_.n_features_in_  # include features in from fitted estimator
        return self  # always return self
    
    def transform(self, X):
        check_is_fitted(self)
        predictions = self.estimator_.predict(X)  # get predictions from estimator_
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)  # if predictions result in a 1d series convert to 2d
        return predictions  # return predictions
    
    def get_feature_names_out(self, names=None):
        check_is_fitted(self)
        n_outputs = getattr(self.estimator_, "n_outputs_", 1)  # need to write output in general way based on what estimator may have
        return [f"Neighbour {i} similarity" for i in range(n_outputs)]

k_neighbors_regressor = KNeighborsRegressor(n_neighbors=5)  # get regressor
neighbor_simil = NearestNeighbourSimilarity(k_neighbors_regressor)  # get transformer

# form nearest neighbour transformer
knn_transformer = NearestNeighbourSimilarity(k_neighbors_regressor)
geo_features = housing[["latitude", "longitude"]]

# short way to check if compiles
check_estimator(NearestNeighbourSimilarity(k_neighbors_regressor))

# print result
print(knn_transformer.fit_transform(geo_features, housing_labels))

# print output features
print(knn_transformer.get_feature_names_out())


preprocessing_new = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),  # divide total bedrooms by total rooms
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),  # divide total rooms by number of households
    ("people_per_house", ratio_pipeline(), ["population", "households"]),  # divide population size by number of houses
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
    "households", "median_income"]),  # apply logarithm to fix large tails
    ("geo", knn_transformer, ["latitude", "longitude"]),  # apply clustering algorithm to see density
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),  # pass through categorical pipeline
],
remainder=default_num_pipeline) # one column remaining: housing_median_age

test_pipeline = make_pipeline(preprocessing_new, SVR(C=10000))
housing_prepared = preprocessing.fit_transform(housing_data, housing_labels)
print(housing_prepared.shape)






# # 5.

# new_k_nearest_neighbours = KNeighborsRegressor(n_neighbors=5)

# new_param_grid = {
#     "preprocessing_new__geo__estimator": [new_k_nearest_neighbours],
#     "svr__C": C
# }


# full_svm_pipeline = Pipeline([
#     ("preprocessing_new", preprocessing_new),
#     ("svr", SVR())
# ])
# new_grid_search = RandomizedSearchCV(estimator=full_svm_pipeline,
#                                      param_distributions=new_param_grid,
#                                      n_iter=50,
#                                      cv=3,
#                                      scoring="neg_root_mean_squared_error",
#                                      error_score="raise")

# new_grid_search.fit(housing_shrunk, housing_labels_shrunk)
# print(f"best parameters: {new_grid_search.best_params_}")
# print(f"best score: {new_grid_search.best_score_}")





# 6.

class StandardScalerClone(BaseEstimator, TransformerMixin):  # input BaseEstimator and TrasnformerMixin for default constructions get_params, set_params as well as fit_transform
    def __init__(self, with_mean=True):  # no *args or **kwargs
        self.with_mean = with_mean

    # implementation of fit
    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)  # sets mean of self
        self.scale_ = X.std(axis=0)  # sets scale of self
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self
    
    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X/self.scale_
    

class NewStandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean  # check if subtract by mean is wanted
        self.with_std = with_std  # check if divide by std is wanted

    def fit(self, X, y=None):
        X_ = X  # copy in order to check if X is dataframe later
        X_ = check_array(X_)  # checks that X_ is an array with finite float values
        self.n_features_in_ = X_.shape[1]  # sets n_features_in_ to number of columns
        if isinstance(X, pd.DataFrame):  # solution is more general if we use hasattr
            self.feature_names_in_ = np.array(X.columns)  # get feature names
        if self.with_mean:  # check if with_mean is true
            self.mean_ = X_.mean(axis=0)
        if self.with_std:  # check if with_std is true
            self.std_ = X_.std(axis=0)
        return self
    
    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes with trailing _
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]  # ensures that n_features_in_ matches number of columns
        if self.with_mean:
            X = X - self.mean_  # subtracts mean
        if self.with_std:
            X = X / self.std_  # divides by std
        return X
    
    def inverse_transform(self, X):  # define inverse transformation
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_std:
            X = X*self.std_
        if self.with_mean:
            X = X + self.mean_
        return X
        
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features != None:
            if len(input_features) == self.n_features_in_:
                if hasattr(self, "feature_names_in_") and not np.all(
                    self.feature_names_in_ == input_features
                ):
                    raise ValueError("input_features ≠ feature_names_in_")
            else:
                raise ValueError("Invalid number of features")
        else:
            return getattr(self,
                           "feature_names_in_",
                           np.array([f"x{i}" for i in range(self.n_features_in_)]))







    
X_new = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "y"])
print(X_new)

# get new scaler
scaler = NewStandardScalerClone(with_mean=True, with_std=True)

# fit data
scaler.fit(X_new)
X_new_transformed = scaler.transform(X_new)
print(f"old:\n{X_new}\nnew:\n{X_new_transformed}")

X_new_new_transformed = scaler.inverse_transform(X_new_transformed)
print(X_new_new_transformed)
print("\n")

print(scaler.get_feature_names_out())






























