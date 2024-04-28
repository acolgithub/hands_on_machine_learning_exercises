



# Q1: What is the fundamental idea behind support vector machines?

print(f"""The fundamental idea behind support vector machines is to find a
      separating hyperplane (separating one particular classification from another)
      which maximizes the distance between the hyperplane ane the nearest points
      on either side. That is, the hyperplane has the largest possible margin.\n""")



# Q2: What is a support vector?

print(f"""A support vector is a vector which lies on the margin and supports
      the plane.\n""")



# Q3: Why is it important to scale the inputs when using SVMs?

print(f"""It is important to scale inputs when using SVMs since determining
      the appropriate hyperplane is affected by scale. In particular,
      the slope may be affected if one feature takes on much larger values.\n""")



# Q4: Can an SVM classifier output a confidence score when it classifies an
#     instance? What about a probability?

print(f"""SVM classifiers can ouput a decision score using the decision function.
      This gives you the signed distance to the decision boundary,
      However, if you use the SVC class you can set the probability
      hyperparameter to True and map the decision function scores to
      estimated probabilities.\n""")



# Q5: How can you choose between LinearSVC, SVC, and SGDClassifier?

print(f"""If you expect you need nonlinear classification and the number
      of training instances is not too large then SVC would be a better
      choice. If you have a large training set it is better to use LinearSVC
      or SGD Classifier. If the training set does not fit in RAM then it is better
      to use SGD Classifier which only uses a little memory.\n""")



# Q6: Say you've trained an SVM classifier with an RBF kernel, but it seems
#     to underfit the training set. Should you increase or decrease γ (gamma)?
#     What about C?

print(f"""If the SVM Classifier has underfit the training set you should
      increase gamma. To fix overfitting you should increase C.\n""")



# Q7: What does it mean for a model to be ϵ-insensitive?

print(f"""A model is epsilon insensitive if adding more instances within
      the margin does not affect the model's predictions.\n""")



# Q8: What is the point of using the kernel trick?

print(f"""The point of using the kernel trick is to simulate using more
      features which encode higher powers of the original feature without
      actually adding them. This permits you to classify nonlinear problems
      without adding more complexity.\n""")



# Q9: Train a LinearSVC on a linearly separable dataset. Then train an SVC
#     and a SGDClassifier on the same dataset. See if you can get them to
#     produce roughly the same model.

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# generate data
p1 = np.array([1,1])
p2 = np.array([-1,-1])
mean = [0,0]
cov = [[1e-1, 0], [0, 1e-1]]
np.random.seed(42)
first_class = p1 + np.random.multivariate_normal(mean=mean, cov=cov, size=100)
second_class = p2 + np.random.multivariate_normal(mean=mean, cov=cov, size=100)

x_first_class, y_first_class = first_class[:,0], first_class[:,1]
x_second_class, y_second_class = second_class[:,0], second_class[:,1]

fig = plt.figure(figsize=(6.5,5.5))
plt.scatter(x_first_class, y_first_class, color="r", marker="^")
plt.scatter(x_second_class, y_second_class, color="b")
plt.xlabel(r"$x$", fontsize=14)
plt.ylabel(r"$y$", fontsize=14)
plt.grid()
plt.savefig("figures/linear_svc_example.png")
plt.close()


svc_linear_clf = make_pipeline(
    StandardScaler(),
    LinearSVC(penalty="l2", loss="hinge", C=5, random_state=42)
)

svc_clf = make_pipeline(
    StandardScaler(),
    SVC(C=5, kernel="linear", random_state=42)
)

sgd_clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss="hinge", penalty="l2", alpha=0.01, max_iter=10000)
)


X = np.concatenate((first_class, second_class))
y1 = np.tile(1, [100])
y2 = np.tile(0, [100])
y = np.concatenate((y1, y2))

indices = np.arange(200)
np.random.shuffle(indices)
X = X[indices,:]
y = y[indices]

svc_linear_clf.fit(X,y)
svc_clf.fit(X,y)
sgd_clf.fit(X,y)

fig = plt.figure(figsize=(6.5,5.5))
plt.scatter(x_first_class, y_first_class, color="r", marker="^")
plt.scatter(x_second_class, y_second_class, color="b")
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
        svc_linear_clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[0],
        alpha=0.5,
        linestyles=["--"]
    )
DecisionBoundaryDisplay.from_estimator(
        svc_clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="m",
        levels=[0],
        alpha=0.5,
        linestyles=[":"]
    )
DecisionBoundaryDisplay.from_estimator(
        sgd_clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="g",
        levels=[0],
        alpha=0.5,
        linestyles=["-"]
    )
plt.xlabel(r"$x$", fontsize=14)
plt.ylabel(r"$y$", fontsize=14)
plt.grid()
plt.savefig("figures/classifier_comparison.png")
plt.close()



scaler = StandardScaler()
scaler.fit(X,y)
X_fit = scaler.transform(X)

lin_clf = LinearSVC(loss="hinge", penalty="l2", C=1, random_state=42)
svc2_clf = SVC(C=1, kernel="linear", random_state=42)
sgd2_clf = SGDClassifier(loss="hinge", penalty="l2", alpha=0.01, max_iter=10000)


lin_clf.fit(X_fit,y)
svc2_clf.fit(X_fit,y)
sgd2_clf.fit(X_fit,y)


def compute_decision_boundary(model):
    w = -model.coef_[0, 0] / model.coef_[0, 1]
    b = -model.intercept_[0] / model.coef_[0, 1]
    return scaler.inverse_transform([[-10, -10 * w + b], [10, 10 * w + b]])

lin_line = compute_decision_boundary(lin_clf)
svc_line = compute_decision_boundary(svc2_clf)
sgd_line = compute_decision_boundary(sgd2_clf)

# Plot all three decision boundaries
plt.figure(figsize=(6.5, 5.5))
plt.plot(lin_line[:, 0], lin_line[:, 1], color="k", linestyle="-", label="LinearSVC")
plt.plot(svc_line[:, 0], svc_line[:, 1], color="m", linestyle="--", linewidth=2, label="SVC")
plt.plot(sgd_line[:, 0], sgd_line[:, 1], color="g", linestyle=":", label="SGDClassifier")
plt.scatter(x_first_class, y_first_class, color="r", marker="^")
plt.scatter(x_second_class, y_second_class, color="b")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.xlim(left=-3, right=3)
plt.ylim(bottom=-3, top=3)
plt.grid()
plt.legend()
plt.savefig("figures/classsifier_comparison2.png")
plt.close()


# Q10: Train an SVM classifier on the wine dataset, which you can load using
#      sklearn.datasets.load_wine(). This dataset contains the chemical
#      analyses of 178 wine samples produced by 3 different cultivators: the
#      goal is to train a classification model capable of predicting the cultivator
#      based on the wine's chemical analysis. Since SVM classifiers are binary
#      classifiers, you will need to use one-versus-all to classify all three
#      classes. What accuracy can you reach?

from sklearn.datasets import load_wine
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd

# load dataset
X_wine, y_wine = load_wine(return_X_y=True, as_frame=True)
print(X_wine)
print(y_wine)

# scale dataset
scaler = StandardScaler()
scaler.fit(X_wine)
X_wine_scaled = pd.DataFrame(scaler.transform(X_wine))


# randomize dataset and split into training/test
np.random.seed(42)
indices = np.arange(len(y_wine))
np.random.shuffle(indices)
X_wine_scaled_train, X_wine_scaled_test = X_wine_scaled.loc[indices[:int(0.8*len(y_wine))],:], X_wine_scaled.loc[indices[int(0.8*len(y_wine)):],:]
y_wine_train, y_wine_test = y_wine[indices[:int(0.8*len(y_wine))]], y_wine[indices[int(0.8*len(y_wine)):]]
print(sum(y_wine_train==0), sum(y_wine_train==1), sum(y_wine_train==2))
print(sum(y_wine_test==0), sum(y_wine_test==1), sum(y_wine_test==2))
print("-"*8)
print(sum(y_wine==0), sum(y_wine==1), sum(y_wine==2))

# get parameter distribution
param_distribution = {
    "svc__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300],
    "svc__kernel": ["rbf", "linear"],
    "svc__gamma": [0.01, 0.1, 1, 10, 100]
}

# make a pipeline
wine_pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))

# get cross validation score of pipeline
print(f"cross-val score: {100*cross_val_score(wine_pipeline, X_wine_scaled_train, y_wine_train).mean()}%")

# set up grid search CV
grid_search = GridSearchCV(estimator=wine_pipeline,
                                 param_grid=param_distribution,
                                 cv=3)

# fit grid search
grid_search.fit(X_wine_scaled_train, y_wine_train)

# get best estimator, optimal score, optimal parameters
classifier_opt = grid_search.best_estimator_
opt_score = grid_search.best_score_
opt_params = grid_search.best_params_
print(opt_score)
print(opt_params)
print("\n")

# score on test set
print(f"Optimal score: {100*grid_search.score(X_wine_scaled_test, y_wine_test)}%")
print(f"Optimal params: {opt_params}")
print(f"Optimal estimator: {grid_search.best_estimator_}\n")



# Q11: Train and fine-tune an SVM regressor on the California housing dataset.
#      You can use the original dataset rather than the tweaked version we used
#      in Chapter 2, which you can load using
#      sklearn.datasets.fetch_california_housing(). The targets represent
#      hundreds of thousands of dollars. Since there are over 20,000 instances,
#      SVMs can be slow, so for hyperparameter tuning you should use far
#      fewer instances (e.g., 2,000) to test many more hyperparameter
#      combinations. What is your best model's RMSE?

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

# get dataset
X_house, y_house = fetch_california_housing(return_X_y=True, as_frame=True)

# make targets into discrete classes
# y_house = y_house.apply(lambda x: round(x))

# look for NAs
print(X_house.isna().sum(), "\n")

# get dtypes
print(X_house.dtypes, "\n")

# get correlation and obtain most correlated columns
# corr_house = X_house
# corr_house["result"] = y_house
# corr_house_matrix = corr_house.corr()
# corr_results = corr_house_matrix["result"]
# high_corr_house = corr_house.loc[:, corr_results.abs() > 0.07]
# X_house = high_corr_house.drop("result", axis=1)

# get training/test
X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(X_house,
                                                                            y_house,
                                                                            test_size=0.2,
                                                                            random_state=42)

# parameter grids
param_grid_svr = {
    "svr__kernel": ["rbf", "linear"],
    "svr__gamma": loguniform(0.001, 0.1),
    "svr__C": uniform(1, 10)
}

# # make pipelines
# house_pipeline_lin = make_pipeline(
#     StandardScaler(),
#     LinearSVR(random_state=42, dual=True, max_iter=100000)
# )

house_pipeline_svr = make_pipeline(
    StandardScaler(),
    SVR()
)


# # fit pipelines
# house_pipeline_lin.fit(X_house_train, y_house_train)
# print((-1)*cross_val_score(estimator=house_pipeline_lin,
#                 X=X_house_train,
#                 y=y_house_train,
#                 scoring="neg_root_mean_squared_error"))

# house_pipeline_svr.fit(X_house_train, y_house_train)
# print((-1)*cross_val_score(estimator=house_pipeline_svr,
#                 X=X_house_train,
#                 y=y_house_train,
#                 scoring="neg_root_mean_squared_error"))


# use randomized search for best parameters
house_grid_search_svr = RandomizedSearchCV(estimator=house_pipeline_svr,
                                     param_distributions=param_grid_svr,
                                     n_iter=100,
                                     cv=3,
                                     random_state=42)

house_grid_search_svr.fit(X_house_train[:2000], y_house_train[:2000])

house_classifier_opt = house_grid_search_svr.best_estimator_
house_opt_params = house_grid_search_svr.best_params_
print(house_classifier_opt)
print(opt_params, "\n")

print(f"final score on train set: {(-1)*cross_val_score(house_classifier_opt, X_house_train, y_house_train, scoring='neg_root_mean_squared_error')}")
print(f"Optimal params: {house_opt_params}")
print(f"Optimal estimator: {house_grid_search_svr.best_estimator_}\n")

y_house_pred = house_classifier_opt.predict(X_house_test)
print(f"final rmse: {mean_squared_error(y_house_test, y_house_pred, squared=False)}")
