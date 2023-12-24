from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier  # import decision tree classifier
from sklearn.tree import export_graphviz  # import graphics for decision tree
# from graphviz import Source  # display file in Jupyter notebook
from sklearn.datasets import make_moons  # get moons dataset
from sklearn.tree import DecisionTreeRegressor  # get decision tree regressor
from sklearn.decomposition import PCA  # get principal component analysis
from sklearn.pipeline import make_pipeline  # import to make pipeline
from sklearn.preprocessing import StandardScaler  # import to scale data
import numpy as np


# import data
iris = load_iris(as_frame=True)

# get data and targets
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

# get decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)

# fit to data
tree_clf.fit(X_iris, y_iris)

# vvisualize tree
export_graphviz(
    decision_tree=tree_clf,  # decision tree
    out_file="iris_tree.dot",  # output file name
    feature_names=["petal length (cm)", "petal width (cm)"],  # feature names
    class_names=iris.target_names,  # target names
    rounded=True,
    filled=True
)

# Source.from_file("iris_tree.dot")  # view tree in Jupyter notebook


# Estimating Class Probabilities

# get probabilites on given instance
print(tree_clf.predict_proba([[5, 1.5]]).round(3))

# get prediction
print(tree_clf.predict([[5, 1.5]]))


# The CART Training Algorithm

# Regularization Parameteres

# get data and targets
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

# get classifier and regularized classifier
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)  # set minimum number of samples per leaf created

# fit to data
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)

# compare the models on new data
X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)

# get scores
print(tree_clf1.score(X_moons_test, y_moons_test))
print(tree_clf2.score(X_moons_test, y_moons_test), "\n")



# Regression

# random quadratic data
np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5  # 200 random values between -0.5 and 0.5
y_quad = X_quad**2 + 0.025*np.random.randn(200, 1)  # 200 random values determined by previous set

# get tree regressor
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)

# fit to data
tree_reg.fit(X_quad, y_quad)



# Sensitivity to Axis Orientation

# make pipleine with principal component analysis
pca_pipeline = make_pipeline(StandardScaler(), PCA())

# get rotated data
X_iris_rotated = pca_pipeline.fit_transform(X_iris)

# get classifier
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)

# fit to data
tree_clf_pca.fit(X_iris_rotated, y_iris)

































