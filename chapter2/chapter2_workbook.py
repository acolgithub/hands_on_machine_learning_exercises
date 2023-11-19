################
### Packages
################

## scikit-learn
import sklearn

# transformers
from sklearn.preprocessing import OrdinalEncoder  # to convert categorical data to numerical
from sklearn.preprocessing import OneHotEncoder  # to use one hot encoding
from sklearn.preprocessing import MinMaxScaler  # to rescale data
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.preprocessing import FunctionTransformer  # for custom transformers
from sklearn.compose import ColumnTransformer  # in order to handle numerical and categorical transformations
from sklearn.compose import make_column_selector, make_column_transformer  # in order to pass selector to prevent manual selection of columns
from sklearn.compose import TransformedTargetRegressor  # to quickly do linear regression and conversions on scaled data
from sklearn.base import BaseEstimator, TransformerMixin  # used to create transformer which could fit to data

# imputing
from sklearn.impute import SimpleImputer  # to impute values

# pipeline
from sklearn.pipeline import Pipeline  # in order to learn about pipelines
from sklearn.pipeline import make_pipeline  # make pipeline function

# train test split
from sklearn.model_selection import train_test_split  # to split into train/test sets

# cross validation
from sklearn.model_selection import cross_val_score

# hyperparameter optimizing
from sklearn.model_selection import GridSearchCV  # to optimize hyperparameters
from sklearn.model_selection import RandomizedSearchCV  # to optimize hyperparmaeters

# regressors
from sklearn.linear_model import LinearRegression  # for linear regression
from sklearn.tree import DecisionTreeRegressor  # to try decision tree regressor
from sklearn.ensemble import RandomForestRegressor  # to try random forest regression

# kmeans
from sklearn.cluster import KMeans

# kernel
from sklearn.metrics.pairwise import rbf_kernel  # for gaussian rbf
from sklearn.metrics import mean_squared_error  # to check accuracy of predictions

# array checking
from sklearn.utils.validation import check_array, check_is_fitted


## pandas
import pandas as pd
from pandas.plotting import scatter_matrix  # used to plot all attributes against eachother

## matplotlib
import matplotlib.pyplot as plt

## numpy
import numpy as np

## scipy
from scipy.stats import randint  # to use randomized search CV

## zlib
from zlib import crc32  # for hashing example of train/test split

## personal
from tar_file_opener import load_tgz_data  # data loader


################
### Functions
################


# function to randomly shuffle data (too primitive)
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))  # shuffle indices
    test_set_size = int(len(data) * test_ratio)  # get test set size
    test_indices = shuffled_indices[:test_set_size]  # first portion of indices for test set
    train_indices = shuffled_indices[test_set_size:]  # last portion of indices for train set
    return data.iloc[train_indices], data.iloc[test_indices]

# compute hash function of identifier, include in test set if in certain proportion of max hash
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

# better function to randomly shuffle data
def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]



################
### Notes
################

### Get the data
###


# load dataset
housing = load_tgz_data(tarfile_input = "housing.tgz",
                        tarurl = "https://github.com/ageron/data/raw/main/")


# print first 5 rows
print(housing.head(5))

# get info about dataframe
housing.info()

# get how many different values a categorical variable takes
print(housing["ocean_proximity"].value_counts())

# get statistical summary
print(housing.describe())

# plot histograms of each numerical attribute of dataframe
housing.hist(bins=50, figsize=(12, 8))
plt.savefig("figures/housing_histplot.png")
plt.close()

# split into train and test set
housing_with_id = housing.reset_index()  # adds an index column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# splitting using sklearn
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# the comment regarding 10.7% chance was determined using binomial distribution
from scipy.stats import binom

print(binom.cdf(k=484, n=1000, p=0.511) + 1 - binom.cdf(k=535, n=1000, p=0.511))


# use cut function to decompose median income into suitable intervals
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])

# plot histogram of category vs. number of districts
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.savefig("figures/income_intervals.png")
plt.close()


# use stratified splitter from sklearn to split data
from sklearn.model_selection import StratifiedShuffleSplit

# 10 splits where each respects strata proportions, test size = 20%
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

# split indices in stratified way accordig to income_cat and for test and train indices
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# get stratified train and test sets (respects strata proportions)
strat_train_set, strat_test_set = strat_splits[0]

# get single split using train_test_splt with stratify argument
strat_train_set, strat_test_set = train_test_split(housing,
                                                                       test_size=0.2,
                                                                       stratify=housing["income_cat"],
                                                                       random_state=42)

print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

# drop categorical income column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)



### Explore and Visualize the Data to Gain Insights
###


# make copy of stratified training set
housing = strat_train_set.copy()

# create scatterplot to visualize geographic data
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
plt.savefig("figures/geographic_figure.png")
plt.close()

# modify opacity to see density
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.savefig("figures/geographic_figure2.png")
plt.close()


# use size and colour to indicate density and price respectively
housing.plot(kind="scatter",
             x="longitude",
             y="latitude",
             grid=True,
             s=housing["population"]/100,
             label="population",
             c="median_house_value",
             cmap="jet",
             colorbar=True,
             legend=True,
             sharex=False,
             figsize=(10,7))
plt.savefig("figures/size_heat_plot.png")
plt.close()


# compute correlation between every pair of attributes
corr_matrix = housing.drop("ocean_proximity", axis=1).corr()

# print correlation matrix
print(corr_matrix)

# check how much attributes correlate with median house value ordered in decreasing values
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# use scatter_matrix to plot many attributes against each other
ordered_attributes = pd.DataFrame(corr_matrix["median_house_value"].sort_values(ascending=False))

# get list of highest correlated attributes
highest_corr = ordered_attributes.index[ordered_attributes.median_house_value>0.1].tolist()

# apply scatter matrix plotting
scatter_matrix(housing[highest_corr], figsize=(12,8))
plt.savefig("figures/scatter_matrix_figure.png")
plt.close()

# focus on median income against median house value
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.savefig("figures/highest_correlated_attribute.png")
plt.close()

# create new data combinations that appear useful
housing["rooms_per_house"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_house"] = housing["population"]/housing["households"]


# get correlation matrix of modified dataframe
corr_matrix_modified = housing.drop("ocean_proximity", axis=1).corr()

# print new correlation matrix
print(corr_matrix_modified, "\n")

# sort attributes by descending correlation to median house value
print(corr_matrix_modified["median_house_value"].sort_values(ascending=False))



### Prepare the Data for Machine Learning Algorithms
###

# revert to clean training set before applying transformations
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 3 ways to handle NA values

print(housing.columns)

# removes rows with NAs
housing.dropna(subset=["total_bedrooms"])  # could have done in place if we wanted to keep it

# remove attribute that has NA values
housing.drop("total_bedrooms", axis=1)

# replace NA values with some other value
median = housing["total_bedrooms"].median(skipna=True)
housing["total_bedrooms"].fillna(median)  # we go with this option but we will use scikitlearn

# printed columns twice to ensure we did not drop any
print(housing.columns)

# create imputer
imputer = SimpleImputer(strategy = "median")

# get numerical subset of housing dataframe
housing_num = housing.select_dtypes(include=[np.number])

# fit imputer to data
imputer.fit(housing_num)

# compare imputer statistics values with direct calculation of medians
print(imputer.statistics_)
print(housing_num.median().values)
print(imputer.statistics_ == housing_num.median().values)

# transform all numerical data from housing_num
X = imputer.transform(housing_num)

# X may be array so we get back column names and indices
housing_tr = pd.DataFrame(X,
                          columns=housing_num.columns,
                          index=housing_num.index)


# handling text and categorical attributes

# view a few instances of categorical attribute
housing_cat = housing["ocean_proximity"]
print(housing_cat.head(8))

# apply encoding to categorical attribute
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat.values.reshape(-1,1))  # convenience function to apply fit and transform, needed to reshape in order to work
# note that -1 in reshape puts everything on first dimension since other dimension is 1

# print first few encoded values
print(housing_cat_encoded[:8])

# get list of categories
print(ordinal_encoder.categories_)

# get one hot encoder and apply to categorical data
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat.values.reshape(-1, 1))  # had to reshape again, result is a scipy sparse matrix

# result is a sparse matrix
print(type(housing_cat_1hot))

# convert matrix to dense numpy array
print(housing_cat_1hot.toarray())  # could also have set sparse=False in OneHotEncoder

# get list of categories
print(cat_encoder.categories_)

# use get_dummies to obtain one-hot representation with one binary feature per category
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
print(pd.get_dummies(df_test))

# use cat_encoder on df_test
print(cat_encoder.transform(df_test.values.reshape(-1, 1)))  # one hot encoder remembers which categories it was trained on

# get_dummies will simply generate a new column for unknwon categories
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
print(pd.get_dummies(df_test_unknown))

# but one hot encoder will recognize it is unknown
cat_encoder.handle_unknown = "ignore"  # suppress warnings
print(cat_encoder.transform(df_test_unknown.values.reshape(-1, 1)))  # ignores unknown category



# Feature Scaling and Transformation

# scale data into uniform range by subtracting min and dividing by (max-min)
min_max_scaler = MinMaxScaler(feature_range = (-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# scale data by standardizing
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# create new feature using Gaussian RBF to compare median age to specific value
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)  # using exp(-gamma(x-35)^2)  as distance function

# define gaussian function
def gauss(x, mean, lamb):
    return np.exp(-lamb*(x-mean)**2)

# create functitons to plot
x_housing = np.linspace(0, 50, 1000)
g1 = [gauss(x, 35, 0.1) for x in x_housing]
g2 = [gauss(x, 35, 0.03) for x in x_housing]

# set up figure and axes
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 5.5))
ax2 = ax.twinx()  # add secondary axes to control second y-axis

# plot histogram and gaussians
ax.hist(housing[["housing_median_age"]], bins=50)  # make histogram
ax2.plot(x_housing, g1, alpha=1.0, color="b")
ax2.plot(x_housing, g2, alpha=0.8, color="b", linestyle="--")

# set up labels, ticks, and legend
ax.set_xlabel("Housing median age", fontsize=16)  # set x label and font size
ax.set_ylabel("Number of districts", fontsize=16)  # set y label and font size
ax2.set_ylabel("Age similarity", fontsize=16, color="b")  # set y label and font size for secondary axis
ax2.tick_params(axis="y", colors="b")  # colour second ary axis ticks
ax2.legend(labels=["gamma=0.10", "gamma=0.03"], loc=2)  # set legend for gaussian curves

plt.savefig("figures/house_age_hist.png")
plt.close()


# get scaler and fit to housing labels
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# get linear regression model and fit to median income and scaled_labels
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)

# get new data
some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

# make predictions in scaled labels
scaled_predictions = model.predict(some_new_data)

# use inverse transformation to get actual predictions
predictions = target_scaler.inverse_transform(scaled_predictions)

# more convenience to use transformed target regressor to do the above for us
model = TransformedTargetRegressor(regressor=LinearRegression(),  # input a regressor
                                   transformer=StandardScaler())  # input a scaler
model.fit(housing[["median_income"]], housing_labels)  # fit to data
predictions = model.predict(some_new_data)  # make predictions


## Custom Transformers

# create transformer which takes log of data
log_transformer = FunctionTransformer(func=np.log, inverse_func=np.exp)

# take logarithm of population
log_pop = log_transformer.transform(housing[["population"]])


# create Gaussian RBF transformer
rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35]],
                                                   gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])



# custom transformers can be used to combine features
ratio_transformer = FunctionTransformer(lambda X: X[:,[0]]/X[:,[1]])
print(ratio_transformer.transform(np.array([[1., 2.],[3.,4.]])))


# create customer transformer class that is similar to standard scaler

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



# implement class which uses KMeans clusterer in fit() method

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


# use cluster similarity class
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1, random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                           sample_weight = housing_labels)


# print first 3 rows of distances to cluster centers
print(similarities[:3].round(2))


# attempt to recreate image
housing_min = housing_labels.min()
housing_max = housing_labels.max()
new_housing_labels = (housing_labels - housing_min)/(housing_max - housing_min)


new_pop = [np.log(x) for x in housing.population]

fig = plt.figure(figsize=(6.5, 5.5))
plt.scatter(housing["longitude"],
            housing["latitude"],
            c = new_housing_labels,  # probably supposed to be label from kmeans
            cmap="jet",
            s=new_pop)
plt.grid()
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("figures/calfornia_housing_heat_map.png")
plt.close()





# Transformation Pipelines

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),  # first impute
    ("standardize", StandardScaler())  # then standardize
])

# make a pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())  # takes transformers as arguments

# apply fit transform method of pipeline
housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2].round(2))

# get dataframe from output of pipeline
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared,
    columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index
)
print(df_housing_num_prepared)


# get transformer in pipeline
print(type(num_pipeline["simpleimputer"]))

# get numerical attributes
num_attribs = list(housing.select_dtypes(include="number").columns)
print(num_attribs)

# categorical attribs
cat_attribs = [x for x in housing.columns if x not in num_attribs]
print(cat_attribs)

# make categorical pipeline
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# preprocess both numerical and categorical by using column transformer
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# preprocessing using column selector
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include = np.number)),
    (cat_pipeline, make_column_selector(dtype_include = object))
)

# apply column transformer to housing data
housing_prepared = preprocessing.fit_transform(housing)






# build pipline representing discussion up to here

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


# run pipeline
housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)

print(preprocessing.get_feature_names_out())





# Select and Train a Model

# we will use linear regression model
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

# apply to training set to make predictions
housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))  # -2 means rounded to the nearest hundred

# compare with labels
print(housing_labels.iloc[:5].values)

# compute root mean squared error

lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(lin_rmse)


## only comment out if you want to check
##  # takes a long time to compile

# # make decision tree pipeline
# tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))

# # fit model to data
# tree_reg.fit(housing, housing_labels)

# # evaluate on training set
# housing_predictions = tree_reg.predict(housing)
# tree_rmse = mean_squared_error(housing_labels,
#                               housing_predictions,
#                               squared=False)

# # get rmse for tree decision classifier
# print(tree_rmse)  # appears to be overfit



# # Better Evaluation Using Cross-Validation

# # use cross validation to evaluate model
# tree_rmses = cross_val_score(tree_reg,
#                              housing,
#                              housing_labels,
#                              scoring="neg_root_mean_squared_error",
#                              cv=10)

# # get results
# print(pd.Series(tree_rmses).describe())


## only comment out if interested to check
## takes a long time to compile

# # make random forest pipeline
# forest_reg = make_pipeline(preprocessing,
#                            RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
#                                 scoring="neg_root_mean_squared_error",
#                                 cv=10)

# # score model
# print(pd.Series(forest_rmses).describe())








































