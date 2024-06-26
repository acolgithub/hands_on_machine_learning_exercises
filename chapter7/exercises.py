



# Q1: If you have trained five different models on the exact same training data,
#     and they all achieve 95% precision, is there any chance that you can
#     combine these models to get better results? If so, how? If not, why?

print(f"""There is a chance that combining the models gives a better result provided that the errors of the different models are not correlated.
Combining many models may improve accuracy since different models can make differernt mistakes but be corrected by the remaining ones.\n""")



# Q2: What is the difference between hard and soft voting classifiers?

print(f"""In hard voting classifiers the most frequently occuring prediction determines the vote.
In soft voting classifiers the votes of the constituent models are combined to form a prediction by averaging the estimated class probabilities.\n""")



# Q3: Is it possible to speed up training of a bagging ensemble by distributing
#     it across multiple servers? What about pasting ensembles, boosting
#     ensembles, random forests, or stacking ensembles?

print(f"""Yes, it is possible to speed up training of a bagging ensemble by distributing it across multiple servers.
This also works for pasting and random forest.
Random forests are trained using the bagging method.
This will not work for boosting ensembles since models are trained on the mistakes of earlier trained models which cannot be done in parallel.
It can work for stacking ensembles.
However, training of one layer will necessarily occur after training another layer (this part cannot be done in parallel).\n""")



# Q4: What is the benefit of out-of-bag evaluation?

print(f"""The benefit of out-of-bag evaluation is that you obtain a validation set without the need for setting aside one earlier.
That is, it is provided by the method.\n""")



# Q5: What makes extra-trees ensembles more random than regular random
#     forests? How can this extra randomness help? Are extra-trees classifiers
#     slower or faster than regular random forests?

print(f"""Extra-trees ensembles are more random than regular random forests since they also choose the threshold for splitting randomly.
This allow for more randomness to be worked into the algorithm.
This helps the alogrithm helps by training the random forest faster.
The algorithm is faster to train since it does not search for the best threshold for splitting.\n""")



# Q6: If you AdaBoost ensemble underfits the training data, which
#     hyperparameters should you tweak, and how?

print(f"""You could try increasing the number of estimators or try weakening the amount of regularizing.\n""")



# Q7: If your gradient boosting ensemble overfits the training set, should you
#     increase or decrease the learning rate?

print(f"""If the gradient boosting ensemble overfits the training set you should increase the learning rate.\n""")



# Q8: Load the MNIST dataset (introduced in Chapter 3), and split it into a
#     training set, a validation set, and a test set (e.g., use 50,000 instances for
#     training, 10,000 for validation, and 10,000 for testing). Then train
#     various classifiers, such as a random forest classifier, an extra-trees
#     classifier, and an SVM classifier. Next, try to combine them into an
#     ensemble that outperforms each individual classifier on the validation
#     set, using soft or hard voting. Once you have found one, try it on the test
#     set. How much better does it perform compared to the individual
#     classifiers?

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


# data
mnist = fetch_openml("mnist_784", as_frame=True)
X_mnist, y_mnist = mnist.data, mnist.target

# scale data
scaler = StandardScaler()
scaler.fit(X_mnist)# 

# get train, validation, and test sets
X_mnist_train, X_mnist_val, X_mnist_test = X_mnist[:50000], X_mnist[50000:60000], X_mnist[60000:]
y_mnist_train, y_mnist_val, y_mnist_test = y_mnist[:50000], y_mnist[50000:60000], y_mnist[60000:]

# get various models
mnist_forest = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42,)
mnist_extra_trees = ExtraTreesClassifier(n_estimators=100, criterion="gini", random_state=42)
mnist_linear_svc = LinearSVC(penalty="l2", loss="squared_hinge", C=3, random_state=42)
# mnist_svc = SVC(kernel="linear", C=3, gamma=1, probability=True, random_state=42)

# # make voting classifiers
# mnist_voting_hard = VotingClassifier(
#     estimators=[
#         ("rf", mnist_forest),
#         ("et", mnist_extra_trees),
#         ("lin_svc", mnist_linear_svc)
#     ],
#     voting="hard"
# )

# mnist_voting_soft = VotingClassifier(
#     estimators=[
#         ("rf", mnist_forest),
#         ("et", mnist_extra_trees),
#         ("svc", mnist_svc)
#     ],
#     voting="soft"
# )

# train all classifiers
mnist_forest.fit(X_mnist_train, y_mnist_train)
mnist_extra_trees.fit(X_mnist_train, y_mnist_train)
mnist_linear_svc.fit(X_mnist_train, y_mnist_train)
mnist_svc.fit(X_mnist_train, y_mnist_train)

mnist_voting_hard.fit(X_mnist_train, y_mnist_train)
mnist_voting_soft.fit(X_mnist_train, y_mnist_train)


# check accuracy of all classifiers
print(f"Accuracy of classifiers")
print(f"random forest: {mnist_forest.score(X_mnist_val, y_mnist_val)}")  # accuracy: 0.9736
print(f"extra trees: {mnist_extra_trees.score(X_mnist_val, y_mnist_val)}")  # accuracy: 0.9743
print(f"linear svc: {mnist_linear_svc.score(X_mnist_val, y_mnist_val)}")  # accuracy 0.8801
print("\n")

y_mnist_val_encoded = y_mnist_val.astype(np.int64)  # convert class names to integers
for estimator in mnist_voting_hard.estimators_:
    print(f"accuracy: {estimator.score(X_mnist_val, y_mnist_val_encoded)}")  # same scores
print("\n")
# print(f"svc: {mnist_svc.score(X_mnist_val, y_mnist_val)}")
print(f"voting hard: {mnist_voting_hard.score(X_mnist_val, y_mnist_val)}")  # with svm accuracy: 0.9741 | without svm accuracy: 0.9735
# print(f"voting soft: {mnist_voting_soft.score(X_mnist_val, y_mnist_val)}")
print("\n")


# evaluate on test set
print(f"Accuracy on test set")
# print(f"random forest: {mnist_forest.score(X_mnist_test, y_mnist_test)}")  # accuracy: 0.968
# print(f"extra trees: {mnist_extra_trees.score(X_mnist_test, y_mnist_test)}")  # accuracy: 0.9703
# print(f"linear svc: {mnist_linear_svc.score(X_mnist_test, y_mnist_test)}")  # accuracy: 0.8797
print("\n")

y_mnist_test_encoded = y_mnist_test.astype(np.int64) 
for estimator in mnist_voting_hard.estimators_:
    print(f"accuracy: {estimator.score(X_mnist_test, y_mnist_test_encoded)}")  # same scores
print("\n")
# print(f"svc: {mnist_svc.score(X_mnist_test, y_mnist_test)}")
print(f"voting hard: {mnist_voting_hard.score(X_mnist_test, y_mnist_test)}")  # with svm accuracy: 0.9682 | without svm accuracy 0.9691
# # print(f"voting soft: {mnist_voting_soft.score(X_mnist_test, y_mnist_test)}")



# Q9: Run the individual classifiers from the previous exercise to make
#     predictions on the validation set, and create a new training set with the
#     resulting predictions: each training instance is a vector containing the set
#     of predictions from all your classifiers for an image, and the target is the
#     image's class. Train a classifier on this new training set. Congratulations
#     --you have just trained a blender, and together with the classifiers it
#     forms a stacking ensemble! Now evaluate the ensemble on the test set.
#     For each image in the test set, make predictions with all your classifiers,
#     then feed the predictions to the blender to get the ensemble's
#     predictions. How does it compare to the voting classifier you trained
#     earlier? Now try again using a StackingClassifier instead. Do you get
#     better performance? If so, why?

# make predictions
y_mnist_forest_pred = mnist_forest.predict(X_mnist_val)
y_mnist_extra_trees_pred = mnist_extra_trees.predict(X_mnist_val)
y_mnist_linear_svc_pred = mnist_linear_svc.predict(X_mnist_val)

# stack predictions
y_mnist_pred = np.array([y_mnist_forest_pred, y_mnist_extra_trees_pred, y_mnist_linear_svc_pred]).T

# blender classifier
blender = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42)

# train on predictions
blender.fit(y_mnist_pred, y_mnist_val)

# evaluate on test set
y_mnist_forest_test_pred = mnist_forest.predict(X_mnist_test)
y_mnist_extra_trees_test_pred = mnist_extra_trees.predict(X_mnist_test)
y_mnist_linear_svc_test_pred = mnist_linear_svc.predict(X_mnist_test)

y_mnist_test_pred = np.array([y_mnist_forest_test_pred, y_mnist_extra_trees_test_pred, y_mnist_linear_svc_test_pred]).T
print(f"blender accuracy: {blender.score(y_mnist_test_pred, y_mnist_test)}")  # accuracy: 0.9693


from sklearn.ensemble import StackingClassifier

# get stacker
stacker = StackingClassifier(estimators=[
    ("fr",mnist_forest),
    ("tr", mnist_extra_trees),
    ("ls", mnist_linear_svc)
],
final_estimator=RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42))

# fit to data
stacker.fit(X_mnist_val, y_mnist_val)

# predict on test set
print(f"stacker: {stacker.score(X_mnist_test, y_mnist_test)}")  # accuracy: 0.9593

print(f"Stacking Classifier achieves lower accuracy on test set. ")




















