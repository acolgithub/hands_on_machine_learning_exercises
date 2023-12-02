from sklearn.preprocessing import add_dummy_feature  # dummy variable for bias
from sklearn.linear_model import LinearRegression  # perform linear regression
from sklearn.linear_model import SGDRegressor  # import stochastic gradient descent regressor
from sklearn.preprocessing import PolynomialFeatures  # import polynomial features
from sklearn.model_selection import learning_curve  # import learning curves to detect overfitting and underfitting
from sklearn.pipeline import make_pipeline  # imported to make pipeline
from sklearn.linear_model import Ridge  # import ridge regression
from sklearn.linear_model import Lasso  # import lasso regression
from sklearn.linear_model import ElasticNet  # import elastic net regression
from sklearn.datasets import load_iris  # load iris dataset
from sklearn.linear_model import LogisticRegression  # import logistic regression
from sklearn.model_selection import train_test_split  # import to split dataset into train and test sets

# for early stopping
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle  # for shuffling indices



# Chapter 4 Training Models


np.random.seed(42)  # to make this code example reproducible
m = 100  # number of instances
X = 2 * np.random.rand(m, 1)  # column vector
y = 4 + 3 * X + np.random.randn(m, 1)  # column vector

# plot linear noise function
fig = plt.figure(figsize=(6.5, 5.5))
plt.scatter(x=X, y=y, color="b")
plt.xlabel("X1", fontsize=14)
plt.ylabel("y", rotation=0, fontsize=14)
plt.xlim(left=0.0, right=2.00)
plt.ylim(bottom=0, top=15)
plt.grid()
plt.savefig("figures/linear_plot.png")
plt.close()


# compute optimal parameters for linear regression
X_b = add_dummy_feature(X)  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# pritn best theta
print(f"Best theta values: {theta_best}\n")

# make predictions using best theta
X_new = np.array([[0], [2]])  # new data
X_new_b = add_dummy_feature(X_new)  # add x0 = 1 to each instance
y_predict = X_new_b @ theta_best  # linear model prediction
print(f"Prediction: {y_predict}\n")  # print prediction


# plot model's predictions (linear regression line)
fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(X_new, y_predict, color="r", linestyle="-", label="Predictions")
plt.scatter(X, y, color="b")
plt.xlabel("X1", fontsize=14)
plt.ylabel("y", rotation=0, fontsize=14)
plt.xlim(left=0.00, right=2.00)
plt.ylim(bottom=0, top=15)
plt.grid()
plt.legend(loc=2)
plt.savefig("figures/linear_regression_predictions.png")
plt.close()


# perform linear regression
lin_reg = LinearRegression()  # get linear regression
lin_reg.fit(X, y)  # fit to data

# get intercept and coefficients
print(f"Linear Regression Intercept: {lin_reg.intercept_}\nLinear Regression Coefficients: {lin_reg.coef_}")

# get prediction
print(f"Linear regression prediction: {lin_reg.predict(X_new)}\n")


# call linear regression directly from numpy
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(f"Optimal theta by direct call to least squares: {theta_best_svd}")


# compute pseudoinverse directly
print(f"Pseudoinverse calculation: {np.linalg.pinv(X_b) @ y}")



# implementation of gradient descent
eta = 0.1  # learning rate, step size
n_epochs = 1000  # number of iterations
m = len(X_b)  # number of instances

np.random.seed(42)
theta = np.random.randn(2, 1)  # randomly initialized model parameters

# iteratively update gradient and theta
for epoch in range(n_epochs):
    gradients = (2/m) * X_b.T @ (X_b @ theta - y)  # explicit gradient of squared loss function
    theta = theta - eta * gradients  # update theta by stepping

# get close to optimal theta
print(f"Near optimal theta: {theta}\n")




# implementation of stochastic gradient descent

n_epochs = 50  # number of iterations
t0, t1 = 5, 50  # learning schedule hyperpameters (parameters to help decrease the step size over time)

# function to slowly decrease step size
def learning_schedule(t):
    return t0/(t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):  # iterate through 50 epochs
    for iteration in range(m):  # for each epoch iterate m times
        random_index = np.random.randint(m)  # select random index
        xi = X_b[random_index:random_index + 1]  # select random row
        yi = y[random_index:random_index+1] 
        gradients = 2*xi.T @ (xi @ theta - yi)  # for SGD do not divide by m, update oarticular partial derivative
        eta = learning_schedule(epoch * m + iteration)  # update learning schedule
        theta = theta - eta * gradients  # update step size


# get theta after apply sgd algorithm
print(f"SGD theta: {theta}\n")


sgd_reg = SGDRegressor(max_iter=1000,  # number of epochs
                       tol=1e-5,  # tolerance in approaching zero gradient 
                       penalty=None,  # no penalty
                       eta0=0.01,  # initial learning rate and uses default learning schedule
                       n_iter_no_change=100,  # algorithm stops if it goes 100 epochs with change less than tolerance
                       random_state=42)


sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets

# results of sgd
print(f"SGD intercept: {sgd_reg.intercept_}\nSGD coefficient: {sgd_reg.coef_}\n")



# Polynomial Regression
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = (1/2) * X**2 + X + 2 + np.random.randn(m, 1)

fig = plt.figure(figsize=(6.5, 5.5))
plt.scatter(x=X, y=y, color="b")
plt.xlabel("X1", fontsize=14)
plt.ylabel("y", rotation=0, fontsize=14)
plt.xlim(left=-3, right=3)
plt.ylim(bottom=0, top=10)
plt.grid()
plt.savefig("figures/polynomial_regression.png")
plt.close()


# input polynomial features of degree 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)  # transform X to include square features

# X is the same
print(X[0], "\n")

# X_poly includes second order (square) and original feature
print(X_poly[0], "\n")



# fit linear regression model to this extended training data
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(f"Linear Regression intercept: {lin_reg.intercept_}\nLinear Regression coefficients: {lin_reg.coef_}\n")


theta_best = [lin_reg.coef_[0,0], lin_reg.coef_[0,1], 1]
xvars = np.linspace(-3, 3, 1000)
xvars_sq = [[x, x**2, lin_reg.intercept_] for x in xvars]
yvars = [xvars_sq[i][0]*theta_best[0] + xvars_sq[i][1]*theta_best[1] + xvars_sq[i][2]*theta_best[2] for i in range(len(xvars))]


fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(xvars, yvars, color="r", label="Predictions")
plt.scatter(X, y, color="b")
plt.xlabel("X1", fontsize=14)
plt.ylabel("y", rotation=0, fontsize=14)
plt.xlim(left=-3, right=3)
plt.ylim(bottom=0, top=10)
plt.grid()
plt.legend(loc=2)
plt.savefig("figures/polynomial_regression_fit.png")
plt.close()




# Learning Curves

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)  # training scores (negative since we use negative scorer)
valid_errors = -valid_scores.mean(axis=1)  # validation scores

fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(train_sizes, train_errors, color="r", linestyle="-", marker="+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, color="b", linestyle="-", linewidth=3, label="valid")
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.xlim(left=0, right=80)
plt.ylim(bottom=0.0, top=2.5)
plt.grid()
plt.legend()
plt.savefig("figures/learning_curve.png")
plt.close()




# polynomial regression pipeline
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),  # add degree 10 features to data
    LinearRegression())  # fit tenth degree polynomial to data

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

# plot training and validation errors against train sizes
fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(train_sizes, train_errors, color="r", marker="+", linestyle="-", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, color="b", linestyle="-", linewidth=3, label="valid")
plt.xlabel("Training set size", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.xlim(left=0, right=80)
plt.ylim(bottom=0.0, top=2.5)
plt.grid()
plt.legend()
plt.savefig("figures/learning_curve2.png")
plt.close()



# ridge regression
ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
print(f"Ridge regression prediction: {ridge_reg.predict([[1.5]])}")


# using stochastic gradient descent instead of direct minimization
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1/m, tol=None,  # added l2 regularization term, changed alpha since algorithm does not average the regularization term
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
print(f"SGD minization instead of direct minimization prediction{sgd_reg.predict([[1.5]])}\n")



# Lasso Regression

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print(f"Lasso Regression prediction: {lasso_reg.predict([[1.5]])}\n")



# Elastic Net Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print(f"Elastic Net Regression prediction: {elastic_net.predict([[1.5]])}\n")




# Early Stopping

# shuffle indices and make training and validation sets
indices = list(range(len(X)))
shuffle(indices)
X_train, y_train, X_valid, y_valid = X[indices[:60]], y[indices[:60]], X[indices[60:]], y[indices[60:]] 

# make preprocessing pipeline
preprocessing = make_pipeline(
    PolynomialFeatures(degree=90, include_bias=False),
    StandardScaler()
)

# preprocess training and validation sets
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)

# introduce SGD regressor with no penalty and initial learning rate 0.002
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500  # number of epochs
best_valid_rmse = float("inf")


for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train.ravel())  # use partial fit to implement incremental learning
    y_valid_predict = sgd_reg.predict(X_valid_prep)  # make prediction
    val_error = mean_squared_error(y_valid, y_valid_predict, squared=False)  # evaluate error on prediction
    if val_error < best_valid_rmse:  # if better than best rmse seen so far then update best rmse and save copy of regression model
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)




# Logistic Regression

# load iris dataset as dataframe
iris = load_iris(as_frame=True)

# print dataset
print(list(iris))
print(iris.data.head(3), "\n")

# print target
print(iris.target.head(3))
print(iris.target_names)

# split dataset into features and target
X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == "virginica"
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# get logistic regression
log_reg = LogisticRegression(random_state=42)

# fit to data
log_reg.fit(X_train, y_train)


# look at model's estimated probabilities for flower with petal widths varying from 0cm to 3cm
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # reshape to get a column vector
y_proba = log_reg.predict_proba(X_new)  # make predictions
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(X_new, y_proba[:, 0], color="b", linestyle="--", linewidth=2,
         label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], color="g", linestyle="-", linewidth="2",
         label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], color="k", linestyle=":",
         linewidth=2, label="Decision boundary")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.xlim(left=0.0, right=3.0)
plt.ylim(bottom=0.0, top=1.0)
plt.grid()
plt.legend(loc=6)
plt.savefig("figures/probabilities_petal_width.png")
plt.close()



# decision boundary value
print(decision_boundary, "\n")

# make predictions near the boundary
print(log_reg.predict([[1.7], [1.5]]), "\n")



# Softmax Regression

# get data and targets
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# get softmax regression (default of logisitic when trained on two or more classes)
softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

# make predictions
print(softmax_reg.predict([[5, 2]]), "\n")

# get prediction probabilities
print(softmax_reg.predict_proba([[5, 2]]).round(2))














