from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier  # get random forest classifier
from sklearn.ensemble import VotingClassifier  # get voting classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier  # get bagging classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  # get accuracy score
from sklearn.ensemble import RandomForestClassifier  # import random forest classifier
from sklearn.ensemble import AdaBoostClassifier  # import ada boost classifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor  # import gradient boosting regressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor  # import hist gradient boost regressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import StackingClassifier  # import stacking c;assifier
import numpy as np





# Voting Classifiers

# get data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# get voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(random_state=42))
    ]
)

# fit classifier to data
voting_clf.fit(X_train, y_train)

# get accuracy of each fitted classifier
for name, clf in voting_clf.named_estimators_.items():
    print(name, "-", clf.score(X_test, y_test))
print("\n")

# make prediction using voting classifier
print(voting_clf.predict(X_test[:1]))

# get individual predictions
print([clf.predict(X_test[:1]) for clf in voting_clf.estimators_])
print("\n")

# get accuracy of voting classifier
print(voting_clf.score(X_test, y_test), "\n")


# set voting parameter to soft
voting_clf.voting = "soft"

# set probability hyperparameter of svc to true
voting_clf.named_estimators["svc"].probability = True

# fit to data
voting_clf.fit(X_train, y_train)

# get accuracy
print(voting_clf.score(X_test, y_test), "\n")




# Bagging and Pasting

bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=500,
                            max_samples=100,
                            oob_score=True,  # get out of bag score
                            n_jobs=-1,  # use all available cores
                            random_state=42)


# fit to data
bag_clf.fit(X_train, y_train)

# get out of bag score
print(bag_clf.oob_score_, "\n")

# get predictions
y_pred = bag_clf.predict(X_test)

# get accuracy
print(accuracy_score(y_test, y_pred), "\n")

# get decision function for first 3 instances (class probabilities)
print(bag_clf.oob_decision_function_[:3], "\n")



# Random Forests
rnd_clf = RandomForestClassifier(n_estimators=500,  # 500 trees
                                 max_leaf_nodes=16,  # each tree limited to 16 leaf nodes
                                 n_jobs=-1,
                                 random_state=42)

# fit to data
rnd_clf.fit(X_train, y_train)

# make prediction
y_pred_rf = rnd_clf.predict(X_test)


# equivalent bagging classifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt",
                           max_leaf_nodes=16),
                           n_estimators=500,
                           n_jobs=-1,
                           random_state=42
)




# Extra-Trees
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier



# Feature Importance

# get data
iris = load_iris(as_frame = True)

# get classifier
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)

# fit to data
rnd_clf.fit(iris.data, iris.target)

# print scores of importance
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)
print("\n")



# Boosting

# AdaBoost

# get classifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=30,
    learning_rate=0.5,
    random_state=42
)

# fit to data
ada_clf.fit(X_train, y_train)



# Gradient Boosting
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:,0]**2 + 0.05*np.random.randn(100)  # y=3x^2 + Gaussian noise

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# train second regressor on residual errors made by first predictor
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)

# train third regressor on residual errors made by second predictor
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)

# new data point
X_new = np.array([[-0.4], [0.], [0.5]])

# make prediction by summing
print(sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3)), "\n")


# same ensemble as previous one
gbrt = GradientBoostingRegressor(max_depth=2,
                                 n_estimators=3,
                                 learning_rate=1.0,
                                 random_state=42)

# fit to data
gbrt.fit(X, y)



# optimal number of trees
gbrt_best = GradientBoostingRegressor(
    max_depth=2,
    learning_rate=0.05,
    n_estimators=500,
    n_iter_no_change=10,
    random_state=42
)

gbrt_best.fit(X, y)

# get number of estimators in optimal version
print(gbrt_best.n_estimators_)



# Histogram-Based Gradient Boosting
hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(),
                            ["ocean_proximity"]),
                            remainder="passthrough"),
    HistGradientBoostingRegressor(categorical_features=[0],
                                  random_state=42)
)
# hgb_reg.fit(housing, housing_labels)



# Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("svc", SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5  # number of cross-validation folds
)

stacking_clf.fit(X_train, y_train)

