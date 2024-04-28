


# Q1: What is the approximate depth of a decision tree trained (without
#     restrictions) on a training set with one million instances?

print(f"""Since the decision tree was trained without restrictions then
the approximate depth of the tree would be log_2(10^6) since we
could repeatedly halve the group until all instances are of the
same type.\n""")


# Q2: Is a node's Gini impurity generally lower or higher than its parent's? Is
#     it generally lower/higher, or always lower/higher?

print(f"""A node's Gini impurity is generally lower than its parent's Gini
impurity. It is generally lower but can be equal or greater.\n""")


# Q3: If a decision tree is overfitting the training set, is it a good idea to try
#     decreasing max_depth?

print(f"""Yes, if the decision tree is overfitting then decreasing max
depth will regularize the decision tree.\n""")


# Q4: If a decision tree is underitting the training set, is it a good idea to try
#     scaling the input features.

print(f"""No, decision trees do not depend on scaling input features.
If it is underfitting the training set this will not help.\n""")


# Q5: If it takes one hour to train a decision tree on a training set containing
#     one million instances, roughly how much time will it take to train
#     another decision tree on a training set containing ten million instances?
#     Hint: consider the CART algorithm's computational complexity.

print(f"""We are given that O(n(10^6)log_2(10^6)) ~ 1 hour. Thus,
O(n(10^7)log_2(10^7)) = 10*O(n(10^6)log_2(10*10^6)) =
10*O(n(10^6)log_2(10^6)) + 10(1/6)*O(n(10^6)log_2(10^6)) ~
10 hours + 10(1/6) hours = 11.6666666667 hours.\n""")


# Q6: If it takes one hour to train a decision tree on a given training set,
#     roughly how much time will it take if you double the number of
#     features?

print(f"""We are given that O(nmlog_2(m)) ~ 1 hour. If we double n then
O(2nmlog_2(m)) = 2*O(nmlog_2(m)) ~ 2 hours.\n""")


# Q7: Train and fine-tune a decision tree for the moons dataset by following
#     these steps:
#
#     a. Use make_moons(n_samples=10000, noise=0.4) to generate a
#        moons dataset.
#     b. Use train_test_split() to split the dataset into a training set and a test
#        set.
#     c. Use grid search with cross-validation (with the help of the
#        GridSearchCV class) to find good hyperparameter values for a
#        DecisionTreeClassifier. Hint: try various values for
#        max_leaf_nodes.
#     d. Train it on the full training set using these hyperparameters, and
#        measure your model's performance on the test set. You should get
#        roughly 85% to 87% accuracy.

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# a.
# get data
X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)

# b.
# split into train/test sets
X_moons_train, X_moons_test, y_moons_train, y_moons_test = train_test_split(X_moons, y_moons, test_size=0.2, random_state=42, shuffle=True)

# c., d.
# get classifier
moons_clf = DecisionTreeClassifier(random_state=42)

# get parameter grid
param_grid = {"decisiontreeclassifier__max_leaf_nodes": [2, 4, 6, 8, 10, 12, 14]}

# make pipeline
moons_pipeline = make_pipeline(moons_clf)

# set up grid search
moons_opt_clf = GridSearchCV(estimator=moons_pipeline, param_grid=param_grid, cv=5)

# find optimal hyperparameters
moons_opt_clf.fit(X_moons_train, y_moons_train)

# get optimal model and parameters
print(f"optimal classifier: {moons_opt_clf.best_estimator_}")
print(f"optimal parameters: {moons_opt_clf.best_params_}")
print(f"optimal score: {moons_opt_clf.best_score_}\n")

# get predictions
pred = moons_opt_clf.predict(X_moons_test)

# get accuracy
print(f"accuracy: {100 * sum(pred == y_moons_test)/len(y_moons_test)}%")
print(f"accuracy function: {100 * accuracy_score(y_moons_test, pred, normalize=True)}%\n")



# Q8: Grow a forest by following these steps:
#
#     a. Continuing the previous exercise, generate 1,000 subsets of the
#        training set, each containing 100 instances selected randomly. Hint:
#        you can use Scikit-Learn's ShuffleSplit class for this.
#     b. Train one decision tree on each subset, using the best
#        hyperparameter values found on the test set. Since they were trained on
#        smaller sets, these decision trees will likely perform worse than the
#        first decision tree, achieving only about 80% accuracy.
#     c. Now comes the magic. For each test set instance, generate the
#        predictions of the 1,000 decision trees, and keep only the most
#        frequent prediction (you can use SciPy's model() function for this).
#        This approach gives you majority-vote predictions over the test set.
#     d. Evaluate these predictions on the test set: you should obtain a
#        slightly higher accuracy than your first model (about 0.5 to 1.5%
#        higher). Congratulations, you have trained a random forest
#        classifier!

from sklearn.model_selection import ShuffleSplit

# a., b.
index_split = ShuffleSplit(n_splits=1000, train_size=0.0125, random_state=42)

pred_arr = []
for train_index, _ in index_split.split(X_moons_train):

    # get decision tree using optimal parameters from earlier
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=12, random_state=42)

    # fit to data
    tree_clf.fit(X_moons_train[train_index,:], y_moons_train[train_index])

    # get predictions
    pred = tree_clf.predict(X_moons_test)

    # print accuracy
    print(f"accuracy: {100 * sum(pred == y_moons_test)/len(y_moons_test)}%")
    print(f"accuracy function: {100 * accuracy_score(y_moons_test, pred, normalize=True)}%\n")

    # store predictions in array
    pred_arr.append(pred)


# c.
import numpy as np
from scipy.stats import mode

pred_arr = np.array(pred_arr)
majority_pred = [mode(pred_arr[:,i]).mode for i in range(len(y_moons_test))]


# d.
print(f"final accuracy: {100 * sum(majority_pred == y_moons_test)/len(y_moons_test)}%")
print(f"final accuracy function: {100 * accuracy_score(y_moons_test, majority_pred, normalize=True)}%")


























