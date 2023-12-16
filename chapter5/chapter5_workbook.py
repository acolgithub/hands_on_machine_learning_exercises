from sklearn.datasets import load_iris  # imported to load iris dataset
from sklearn.pipeline import make_pipeline  # imported to make pipeline
from sklearn.preprocessing import StandardScaler  # imported to scale data
from sklearn.svm import LinearSVC  # imported to apply supprt vector classifier
from sklearn.datasets import make_moons  # dataset with points froming interleaving crescent moons
from sklearn.preprocessing import PolynomialFeatures  # imported to add polynomial features
from sklearn.svm import SVC  # imported to use SVC
from sklearn.inspection import DecisionBoundaryDisplay  # imported to plot decision boundary
from sklearn.svm import LinearSVR  # imported to use linear svr (regression)
from sklearn.svm import SVR  # imported to use SVR  (regression)

import numpy as np
import matplotlib.pyplot as plt

# Linear SVM Classification

# load dataset
iris = load_iris(as_frame=True)

# get data
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # Iris virginica

# make svm pipeline
svm_clf = make_pipeline(StandardScaler(),  # first scale data
                        LinearSVC(C=1, random_state=42))  # then apply linear svm

# fit to data
svm_clf.fit(X, y)

# make predictions
X_new = [[5.5, 1.7], [5.0, 1.5]]
print(f"predictions: {svm_clf.predict(X_new)}\n")

# scores used to make predictions (measures signed distance of instance to decision boundary)
print(f"svm decision function: {svm_clf.decision_function(X_new)}\n")



# Nonlinear SVM Classification

# get data
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# make pipeline which adds polynomial features, scales data, then uses linear SVC
polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42)  # can add underscore to separate zeros
)

# fit to data
polynomial_svm_clf.fit(X,y)



# Polynomial Kernel

# make pipeline with scaler and svc with third degree polynomial kernel
poly_kernel_svm_clf = make_pipeline(StandardScaler(),
                                    SVC(kernel="poly", degree=3, coef0=1, C=5))  # coef0 determines influence on model of high degree terms vs low degree terms
poly_kernel_svm_clf.fit(X,y)



# Similarity Features

# define gaussian function
def gaussian(x, loc=0, gamma=1):
    return np.exp(-gamma*(x-loc)**2)

# get x-values
x = [-4 + i for i in range(9)]

# get gaussian y-values
y1 = [gaussian(t, -2, 0.3) for t in x]
y2 = [gaussian(t, 1, 0.3) for t in x]

# form triples (x, y1(x), y2(x))
triples = np.array([[x[i], y1[i], y2[i]] for i in range(9)])

# get pairs (y1(x), y2(x))
pairs = triples[:,1:]

# get blue and green points
blue_pairs = np.array([s[1:] for s in triples if np.abs(s[0])>=3])
green_pairs = np.array([s[1:] for s in triples if np.abs(s[0])<3])

# classify points
colour_classifier = [np.abs(i-4)>=3 for i in range(9)]

lin_svc_clf = LinearSVC(C=100, random_state=420)
lin_svc_clf.fit(pairs, colour_classifier)

fig = plt.figure(figsize=(13, 5.5))

plt.subplot(121)

x_blue = [x[i] for i in range(9) if colour_classifier[i]]
x_green = [x[i] for i in range(9) if not colour_classifier[i]]
x_vec = np.array(np.linspace(-4.5, 4.5, 1000))
y_vec1 = np.array([gaussian(xi, -2, 0.3) for xi in x_vec])
y_vec2 = np.array([gaussian(xi, 1, 0.3) for xi in x_vec])

plt.scatter(x_blue, np.zeros(len(x_blue)), color="b", marker="s")
plt.scatter(x_green, np.zeros(len(x_green)), color="g", marker="^")
plt.plot(x_vec, y_vec1, color="g", linestyle="--")
plt.plot(x_vec, y_vec2, color="b", linestyle="--")
plt.hlines(y=0.0, xmin=-4.5, xmax=4.5, color="k", linestyle="-", linewidth=2)
plt.scatter([-2, 1], [0, 0], color="r", alpha=0.35, s=100)
plt.xlabel(r"$x_{1}$", fontsize=12)
plt.ylabel(r"Similarity", fontsize=12)
plt.xlim(left=-4.5, right=4.5)
plt.ylim(bottom=-0.0625, top=1.0625)
plt.grid()

plt.subplot(122)
plt.scatter(blue_pairs[:,0], blue_pairs[:,1], color="b", marker="s")
plt.scatter(green_pairs[:,0], green_pairs[:,1], color="g", marker="^")
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
        estimator=lin_svc_clf,
        X=pairs,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="r",
        levels=[0],
        linestyles=["--"],
    )
plt.hlines(y=0.0, xmin=-0.1, xmax=1.1, color="k", linestyle="-", linewidth=2)
plt.vlines(x=0.0, ymin=-0.05, ymax=1.05, color="k", linestyle="-", linewidth=2)
plt.xlabel(r"$x_{2}$", fontsize=12)
plt.ylabel(r"$x_{3}$", fontsize=12)
plt.xlim(left=-0.1, right=1.1)
plt.ylim(bottom=-0.05, top=1.05)
plt.grid()
plt.savefig("figures/svc_similarity_features.png")
plt.close()



# Gaussian RBF Kernel

# use svm with rbf kernel after scaling
rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", gamma=5, C=0.001))
rbf_kernel_svm_clf.fit(X, y)



# SVM Regression

svm_reg = make_pipeline(StandardScaler(),
                        LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(X,y)



svm_poly_reg = make_pipeline(StandardScaler(),
                             SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1))
svm_poly_reg.fit(X,y)















