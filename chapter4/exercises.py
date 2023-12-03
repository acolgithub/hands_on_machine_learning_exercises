# 1.

print(f"In a situation where the training set has millions of features (i.e. large n) you could use the Stochastic Gradient Descent Algorithm.\n")



# 2.

print(f"Gradient Descent algorithms might have difficulty if the features have very different scales. To fix this, you should rescale the data to be in a common range.\n")



# 3.

print(f"The cost function for a logistic regression model is convex so gradient descent cannot get caught in a local minimum.\n")



# 4.

print(f"The algorithms may not converge to the same model if the learning rate is not adjusted. In addition, stochastic algorithms need the learning rate decreased over time in order to converge to the same model (this is to counteract the build-in randomness).\n")



# 5.

print(f"The model is diverging from the minimizer. To fix this, you should decrease the learning rate so that the algorithm can converge to the minimizer.\n")



# 6.

print(f"This is not a good idea since mini-batch gradient descent makes random choices of small batches which could sometimes result in an increase in cost. It would be better to stop the algorithm if this happens consistently for a number of iterations and revert to the minimum.\n")



# 7.

print(f"Stochastic gradient descent will reach the viscinity of the optimal solution the fastest since it only calculates the gradient on a single sample which is quite fast. However, this algorithm will not converge unless you adjust the larning rate since the built-in randomness. Batch gradient descent will actually converge.\n")



# 8.

print(f"In this scenario the model is overfitting to the training data which causes a large gap between training error and validation error. To solve this, since this is polynomial regression, you could the degree of the regression polynomial. You could also regularize the model to reduces it complexity. Another way would be to provide more training data if we are certain of the degree.\n")



# 9.

print(f"Since you are seeing large training/validation errors then the model is underfitting likely has high bias. We should decrease alpha to allow the model to fit more parameters to the data. Increasing the complexity could help reduce the large training/validation errors.\n")



# 10.

# a.
print(f"It is often good to use ridge regression as opposed to plain linear regression.")

# b.
print(f"You would use lasso regression instead of ridge regression if you wanted to eliminate unnecessary weights and removing such weights was of high importance.")

# c.
print(f"You would use elastic net regression instead of lasso regression when you want to penalize unnecessary weights but not too strongly.\n")



# 11.

print(f"Since the number of classes is just 2 for each classification and since you want multiple classifications (the classifications are independent) then you would use two logistic classifiers.\n")



# 12.

import numpy as np
from sklearn.datasets import load_iris

# define softmax function
def softmax(X, Theta):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Theta, np.ndarray):
        Theta = np.array(Theta)
    softmax_scores = X@Theta
    exp_softmax_scores = np.exp(softmax_scores)
    return exp_softmax_scores/exp_softmax_scores.sum(axis=1, keepdims=True)

# assign prediction based on most likely class
def softmax_classifier(X, Theta):
    return np.argmax(softmax(X, Theta), axis=1)

# compute cross entropy cost
def cross_entropy_cost(X, y, Theta, C=np.inf, tol=1e-5):
    log_prob = np.log(softmax(X, Theta)+tol)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    softmax_error = (-1/y.shape[0])*(y.T@log_prob).trace()  # error from softmax
    regularization_error = (1/(2*C))*(Theta[1:]**2).sum()  # error from regularization
    return softmax_error + regularization_error

# compute gradient of cross entropy cost
def grad_cross_entropy_cost(X, y, Theta, C=np.inf):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    softmax_grad = (1/y.shape[0])*X.T@((softmax(X, Theta) - y))  # gradient from softmax
    regularization_grad = np.r_[np.zeros((1,Theta.shape[0])), (1/C)*Theta[1:,:]]  # gradient from regularization
    return softmax_grad + regularization_grad

# perform gradient descent
def batch_gradient_descent(X_train, y_train, X_valid, y_valid,
                           Theta, grad, C=np.inf, eta=0.01, n_epoch=15, n_iter_larger=100):
    
    # set number of iteration validation score increased
    times_larger = 0

    # set minimal observed validation error score
    least_valid_score = 1e10

    # set optimal theta
    Theta_best = Theta

    # iterate to perform gradient descent
    for i in range(n_epoch):

        # compute cross entropy cost on training/validation sets
        current_train_score = cross_entropy_cost(X_train, y_train, Theta, C)
        current_valid_score = cross_entropy_cost(X_valid, y_valid, Theta, C)

        # if smaller update optimal parameters
        if current_valid_score < least_valid_score:
            least_valid_score = current_valid_score
            Theta_best = Theta
            times_larger = 0
        else:
            times_larger += 1  # increment counter of times validation error grew

        # if validation error grew for too long end loop
        if times_larger >= n_iter_larger:
            print(f"\nScore on validation set did not decrease for {n_iter_larger} iterations. Terminating.")
            print(f"Best training score was: {current_train_score}")
            print(f"Best validation score was {least_valid_score}\n")
            return [Theta_best, current_train_score, least_valid_score]
        
        # print current training and validation scores
        print(f"Current training score: {current_train_score}")
        print(f"Current validation score: {current_valid_score}\n")

        # update theta
        Theta = Theta - eta*grad(X_train, y_train, Theta)

    # print optimal training/validation error values
    print(f"Best training score was: {current_train_score}")
    print(f"Best validation score was {least_valid_score}\n")

    return [Theta_best, current_train_score, least_valid_score]

# standardize data
def standardize(X):
    return (X - X.mean(axis=0))/X.std(axis=0)

# add column of ones to data
def add_ones(X):
    return np.c_[np.ones(X.shape[0]), X]

# one hot encode  target
def one_hot_encode(y):
    y_encoded = np.zeros((y.shape[0], 3))
    for i in range(y.size):
        y_encoded[i, y[i]] = 1
    return y_encoded

# split data into training/validation/test sets
def data_split(X, y):

    # set seed for reproducibility
    np.random.seed(42)

    # get number of samples
    m = y.shape[0]

    # get indices and randomize
    indices = np.arange(m)
    np.random.shuffle(indices)

    # set aside 60% for training, 20% for validation, 20% for testing
    train_index = int(m*0.6)
    val_index = train_index + int(m*0.2)

    # split data into training, validation, and testing sets
    X_train, X_valid, X_test = X[indices[:train_index], :], X[indices[train_index:val_index],:], X[indices[val_index:],:]
    y_train, y_valid, y_test = y[indices[:train_index]], y[indices[train_index:val_index]], y[indices[val_index:]]

    return X_train, X_valid, X_test, y_train, y_valid, y_test

# preprocess data
def data_preprocessor(X, y):

    # convert y to indicator of class
    y_ones = one_hot_encode(y)

    # split into train, validation, test sets
    X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(X, y_ones)

    # standardize data
    X_train, X_valid, X_test = standardize(X_train), standardize(X_valid), standardize(X_test)

    # add bias term
    X_train, X_valid, X_test = add_ones(X_train), add_ones(X_valid), add_ones(X_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

# get optimal model
def get_optimal_model(X_train, y_train, X_valid, y_valid,
                           Theta, grad, Cs, etas, n_epoch=15, n_iter_larger=100):
    
    # intialize optimal parameters
    optimal_Theta = Theta
    optimal_eta = etas[0]
    optimal_C = Cs[0]
    optimal_train_score = 1e10
    optimal_valid_score = 1e10

    # loop over gradient descent to find best parameters
    for eta in etas:
        for C in Cs:

            # copy of inputs
            Theta_loc = Theta

            Theta_ouput, train_score, valid_score= batch_gradient_descent(X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        Theta=Theta_loc,
                        grad=grad_cross_entropy_cost,
                        eta=eta,
                        C=C,
                        n_epoch=10000,
                        n_iter_larger=10)
            if valid_score < optimal_valid_score:
                optimal_Theta = Theta_ouput
                optimal_eta = eta
                optimal_C = C
                optimal_train_score = train_score
                optimal_valid_score = valid_score
    return optimal_Theta, optimal_eta, optimal_C, optimal_train_score, optimal_valid_score




# get data
iris = load_iris(as_frame=True)

# load data
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"].values

# preprocess data
X_train, X_valid, X_test, y_train, y_valid, y_test = data_preprocessor(X, y)

# randomly start Theta
Theta = np.random.rand(y_train.shape[1], X_train.shape[1])

# parameters
etas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

Theta_opt, eta_opt, C_opt, train_score_opt, valid_score_opt = get_optimal_model(X_train=X_train,
                  y_train=y_train,
                  X_valid=X_valid,
                  y_valid=y_valid,
                  Theta=Theta,
                  Cs=Cs,
                  grad=grad_cross_entropy_cost,
                  etas=etas,
                  n_epoch=10000,
                  n_iter_larger=10)

print(f"Optimal model Theta: {Theta_opt}")  # optimal theta [[ 0.30978813  3.10947392 -2.00403576] [-3.21422324  0.07332627  3.7364265 ] [-2.9217746  -0.37364666  5.19446058]]
print(f"Optimal model eta: {eta_opt}")  # optimal eta is 10
print(f"Optimal model C: {C_opt}")  # optimal C is 1000 (from testing infinity appears to be the optimal value so regularization is not helping)
print(f"Optimal model training error: {train_score_opt}")  # 0.08316510720931711
print(f"Optimal model validation score: {valid_score_opt}\n")  # optimal validation error 0.20775621582144493
print(f"Optimal model prediction accuracy on training set: {100*(softmax_classifier(X_train, Theta_opt) == np.argmax(y_train, axis=1)).mean()}%")  # scored 95.55555555555556%
print(f"Optimal model prediction accuracy on validation set: {100*(softmax_classifier(X_valid, Theta_opt) == np.argmax(y_valid, axis=1)).mean()}%")  # scored 90.0%
print(f"Optimal model prediction accuracy on test set: {100*(softmax_classifier(X_test, Theta_opt) == np.argmax(y_test, axis=1)).mean()}%")  # scored 96.66666666666667%












