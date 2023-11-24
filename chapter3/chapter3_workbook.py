# sklearn
from sklearn.datasets import fetch_openml  # to get dataset
from sklearn.linear_model import SGDClassifier  # for sgd classifier
from sklearn.model_selection import cross_val_score  # to cross validate
from sklearn.dummy import DummyClassifier  # get dummy classifier which assigns most common class
from sklearn.model_selection import StratifiedKFold  # for stratified samplling
from sklearn.base import clone  # to clone classifier
from sklearn.model_selection import cross_val_predict  # to get cross validation predictions
from sklearn.metrics import confusion_matrix  # get confusion matrix
from sklearn.metrics import precision_score, recall_score  # get precision and recall scores
from sklearn.metrics import f1_score  # get f1 score
from sklearn.metrics import precision_recall_curve  # get precision recall curve
from sklearn.metrics import roc_curve  # get roc curve
from sklearn.metrics import roc_auc_score  # get area under curve score
from sklearn.ensemble import RandomForestClassifier  # get random forest classifier
from sklearn.svm import SVC  # get support vector machine classifier
from sklearn.multiclass import OneVsRestClassifier  # get one vs rest classifier
from sklearn.preprocessing import StandardScaler  # get standard scaler
from sklearn.metrics import ConfusionMatrixDisplay  # get confusion matrix image display
from sklearn.neighbors import KNeighborsClassifier  # get K nearest neighbours classifier
from sklearn.multioutput import ClassifierChain  # get chain classifier for multilabels

# matplotlib
import matplotlib.pyplot as plt

# numpy
import numpy as np

    

# data
mnist = fetch_openml("mnist_784", as_frame=False)  # aset as_frame = False to get numpy arrays



X, y= mnist.data, mnist.target
print(X, "\n")

# digit plot
def plot_digit(image_data, filename):
    image = image_data.reshape(28, 28)  # reshape to be image
    plt.imshow(image, cmap="binary")  # get binary colour map
    plt.axis("off")  # removes axis
    plt.savefig(filename)
    plt.close()

# plot digit
some_digit = X[0]
plot_digit(some_digit, "figures/some_digit_plot.png")  # looks like 5
print(f"first data entry: {X[0]}\n")  # first data entry
print(f"label of first entry: {y[0]}")  # confirm it is 5

# split into train and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]



# Training a Binary Classifier
y_train_5 = (y_train == "5")  # True for all 5s, False for all other digits
y_test_5 = (y_test == "5")

# get classifier
sgd_clf = SGDClassifier(random_state=42)

# fit to data
sgd_clf.fit(X_train, y_train_5)

# predict a digit
print(sgd_clf.predict([some_digit]))  # needs array to be passed, guesses correctly


# Performance Measures
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))  # cross validate with 3 folds


# get dummy classifier
dummy_clf = DummyClassifier()

# train dummy classifier
dummy_clf.fit(X_train, y_train_5)

# make predictions
print(any(dummy_clf.predict(X_train)))  # prints False: no 5s detected since most frequent class is non-5
# checked if there are any 5s

# get cross validation score
print(cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"))



# implementing cross-validation
skfolds = StratifiedKFold(n_splits=3)  # add shuffle=True if dataset is not
                                       # already shuffled

# get train and test data for 3 strata
for train_index, test_index in skfolds.split(X_train, y_train_5):

    # get classifier
    sgd_clf2 = SGDClassifier(random_state=42)

    # clone classifier
    clone_clf = clone(sgd_clf2)

    # split into train and test sets for each strata
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    # fit cloned classifier
    clone_clf.fit(X_train_folds, y_train_folds)

    # make predictions
    y_pred = clone_clf.predict(X_test_fold)

    # print percentage correct
    print(f"percentage predicted correct: {(y_pred == y_test_fold).mean()}")





# back to studying non-stratified

# use cross validation to predict to get predictions made
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# get confusion matrix
cm = confusion_matrix(y_train_5, y_train_pred)  # rows are actual class, columns are predicted class
print(cm)

# example of perfect predictions
y_train_perfect_predictions = y_train_5  # pretend we rearched perfection
print(confusion_matrix(y_train_5, y_train_perfect_predictions))

# print precision and recall
print(f"precision score: {precision_score(y_train_5, y_train_pred)}\n")  # ==3530/(687+3530)
print(f"recall score: {recall_score(y_train_5, y_train_pred)}")  # == 3530/(1891 + 3530)

# print f1 score
print(f"f1 score: {f1_score(y_train_5, y_train_pred)}")

# get decision function on some example and score used to make prediction
y_scores = sgd_clf.decision_function([some_digit])

# print score
print(y_scores)

# set threshold (sgd classifier uses 0 as a threshold as well)
threshold = 0

# make prediction based on threshold
y_some_digit_pred = (y_scores > threshold)
print(f"prediction based on zero threshold: {y_some_digit_pred}")

# raise threshold
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print(f"prediction based on 3000 threshold: {y_some_digit_pred}")

# to decide threshold get cross_val_predict to get scores of all instances
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


# get precision recall curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# get index 
threshold_index = list(thresholds >= threshold).index(True)
threshold_value = thresholds[threshold_index]
precision_thresh = precisions[threshold_index]
recall_thresh = recalls[threshold_index]

# plot thresholds against precisions and recalls to choose threshold
fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(thresholds,
         precisions[:-1],
         color="b",
         linestyle="--",
         label="Precision",
         linewidth=2)
plt.plot(thresholds,
         recalls[:-1],
         color="g",
         linestyle="-",
         label="Recall",
         linewidth=2)
plt.vlines(threshold, 0, 1.0, color="k", linestyle="dotted", label="threshold")
plt.grid()
plt.legend()
plt.xticks([-40000 + 20000*i for i in range(5)])
plt.xlim(left=-50000, right=50000)
plt.ylim(bottom=0.0, top=1.0)
plt.plot(threshold_value, precision_thresh, color="b")
plt.plot(threshold_value, recall_thresh, color="g")
plt.savefig("figures/precision_recall_curves.png")
plt.close()


# plot precisions directly against recalls to choose threshold
fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(recalls,
         precisions,
         linewidth=2,
         label="Precision/Recall curve")
plt.plot(recall_thresh,
         precision_thresh,
         color="k",
         marker="o",
         markersize="5",
         label="Point at threshold 3,000")
plt.vlines(recall_thresh, 0.0, precision_thresh, linestyle="dotted", color="k")
plt.hlines(precision_thresh, 0.0, recall_thresh, linestyle="dotted", color="k")
plt.grid()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(left=0.0, right=1.0)
plt.ylim(bottom=0.0, top=1.0)
plt.legend(loc=3)
plt.savefig("figures/precision_vs_recall.png")
plt.close()


# get index of lowest threshold giving 90% precision
idx_for_90_precision = (precisions >= 0.90).argmax()

# get threshold for 90% precision
threshold_for_90_precision = thresholds[idx_for_90_precision]
print(f"threshold for 90% precision: {threshold_for_90_precision}")

# get predictions manually using threshold and scores from sgd classifier
y_train_pred_90 = (y_scores >= threshold_for_90_precision)

# check these predictions precision and recall
print(f"precision score: {precision_score(y_train_5, y_train_pred_90)}")
print(f"recall score: {recall_score(y_train_5, y_train_pred_90)}")


# get false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# get index of threshold of 90% precision
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()

# get true positive rate and false positive rate at this threshold
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

# plot roc curve
fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0,1], [0,1], color="k", linestyle="dotted", label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], color="k", marker="o", label="Threshold for 90% precision")
plt.grid()
plt.xlabel("False Positive Rate (Fall-Out)")
plt.ylabel("True Positive Rate (Recall)")
plt.xlim(left=0.0, right=1.0)
plt.ylim(bottom=0.0, top=1.0)
plt.legend()
plt.savefig("figures/roc.png")
plt.close()


# get area under curve score for roc curve
print(f"area under curve score: {roc_auc_score(y_train_5, y_scores)}")



# create random forest classifier for comparison
forest_clf = RandomForestClassifier(random_state=42)

# train random forest classifier using cross validation and predict class probabilities
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# look at class probabilities for first two images in training set
print(f"first two class probabilities: {y_probas_forest[:2]}")

# use estimated probabilities for positive class as scores
y_scores_forest = y_probas_forest[:, 1]

# get precision recall curve for random forest classifier
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest
)


# plot precision recall curve for random forest classifier
fig = plt.figure(figsize=(6.5, 5.5))
plt.plot(recalls_forest, precisions_forest, color="b", linestyle="-", label="Random Forest")
plt.plot(recalls, precisions, linestyle="--", linewidth=2, label="SGD")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(left=0.0, right=1.0)
plt.ylim(bottom=0.0, top=1.0)
plt.grid()
plt.legend(loc=3)
plt.savefig("figures/random_forest_precision_recall_curve.png")
plt.close()


# get f1 and roc area under curve scores for random forest classifier
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba >= 50%
print(f"f1 score of random forest classifier: {f1_score(y_train_5, y_train_pred_forest)}")
print(f"roc area under curve score for random forest classifier: {roc_auc_score(y_train_5, y_scores_forest)}\n")


# get precision and recall scores for random forest classifier
print(f"precision score of random forest classifier: {precision_score(y_train_5, y_train_pred_forest)}")
print(f"recall score of random forest classifier: {recall_score(y_train_5, y_train_pred_forest)}")




# Multiclass Classification

# get support vector machine classifier
svm_clf = SVC(random_state=42)

# fit to portion of data (sklearn wil use one vs one strategy to train many binary classifiers on small data)
svm_clf.fit(X_train[:2000], y_train[:2000])  # y_train not y_train_5
# use only 2000 images since it is slow when including large datasets


# make prediction
print(f"support vector machine prediction: {svm_clf.predict([some_digit])}")

# get decision scores used to make prediction
some_digit_scores = svm_clf.decision_function([some_digit])
print(f"decision scores of support vector machine: {some_digit_scores.round(2)}")

# get class id of maximum score
class_id = some_digit_scores.argmax()
print(f"class id of maximum score: {class_id}")

# obtain classes from trained classifier
print(f"target classes: {svm_clf.classes_}")

# get the class at index matching class_id
print(f"class at index=class_id: {svm_clf.classes_[class_id]}")




# create one vs rest classifier where support vector machine classifier passed
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])


# make prediction with one vs rest classifier
print(f"one vs rest prediction: {ovr_clf.predict([some_digit])}")

# check number of trained classifiers
print(f"number of trained classifiers: {len(ovr_clf.estimators_)}")



# train SGD classifier on multilcass dataset to make predictions
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
# print(f"sgd prediction : {sgd_clf.predict([some_digit])}")  # sklearn uses one vs rest strategy


# get sgd classifier assigned scores
print(f"classifier assigned scores: {sgd_clf.decision_function([some_digit]).round()}")


# use cross validation to evaluate sgd
print(f"evaluation score of sgd: {cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')}")


# get standard scaler
scaler = StandardScaler()

# fit and transform training data (scale the pixels)
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

# get cross validation score of scaled training data
print(f"cross validation score of sgd on training set: {cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')}")



# Error Analysis

# get confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.savefig("figures/confusion_matrix.png")
plt.close()

# get normalized confusion matrix (normalized by row or total true labels)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true",
                                        values_format=".0%")
plt.savefig("figures/confusion_matrix_normalized.png")
plt.close()


# put zero weight on correct predictions to emphasize errors
sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true",
                                        values_format=".0%")
plt.savefig("figures/confusion_matrix_normalized_weighted.png")
plt.close()

# zero weight on correct predictions and normalize by column
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="pred",
                                        values_format=".0%")
plt.savefig("figures/confusion_matrix_normalized_weighted.png")
plt.close()



# Multilabel Classification

# create multilabel y
y_train_large = (y_train >= "7")  # labels greater than or equal to 7
y_train_odd = (y_train.astype("int8")%2 == 1)  # odd labels
y_multilabel = np.c_[y_train_large, y_train_odd]

# train classifier on multiple labels
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# get prediction on a digit (now has two outputs since multilabel)
print(f"knn prediction: {knn_clf.predict([some_digit])}")

# evaluate multilabel model using f1 scores on each label
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
print(f"f1 scores averaged across label: {f1_score(y_multilabel, y_train_knn_pred, average='macro')}")
print(f"f1 scores averaged across label and weighted by number of instances of label {f1_score(y_multilabel, y_train_knn_pred, average='weighted')}")



# use chain classifier to allow svc to do multilabel predictions
chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

# make predictions with chain classifier
print(f"chain classifier prediction on digit: {chain_clf.predict([some_digit])}")



# Multioutput Classification

# add noise to MNIST train set
np.random.seed(42)  # to make this code example reproducible
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

# add noise to MNIST test set
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

# the training and test labels are the original images
y_train_mod = X_train
y_test_mod = X_test

# train knn to classify image
knn_clf = KNeighborsClassifier()

# fit to data
knn_clf.fit(X_train_mod, y_train_mod)

# make prediction
clean_digit = knn_clf.predict([X_test_mod[0]])

# plot clean digit
plot_digit(clean_digit, "figures/clean_digit.png")














