from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from keras.utils import plot_model

import keras_tuner as kt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from time import strftime



# data
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target==0)  # Iris setosa

# get perceptron
per_clf = Perceptron(random_state=42)
per_clf.fit(X,y)

# predict on new data
X_new = [[2,0.5], [3,1]]
y_pred = per_clf.predict(X_new)  # predicts True and False for these 2 flowers



# Regression MLPs

# get data
housing = fetch_california_housing()

# split into train, validation, test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# get mlp regressor
mlp_reg = MLPRegressor(hidden_layer_sizes=[50,50,50], random_state=42)

# scale data then apply mlp
pipeline = make_pipeline(StandardScaler(), mlp_reg)

# fit to data
pipeline.fit(X_train, y_train)

# male predictions
y_pred = pipeline.predict(X_valid)
y_pred = pipeline.predict(X_valid)

# evaluate error
rmse = mean_squared_error(y_valid, y_pred, squared=False)  # about 0.505
print(rmse, "\n")


# classification MLPs

# get train and test sets
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X, y, test_size=0.2, random_state=42)

# get mlp classifier
mlp_clf = MLPClassifier(max_iter=500, random_state=42)

# scale data then apply mlp classifier
mlp_clf_pipeline = make_pipeline(StandardScaler(), mlp_clf)
mlp_clf_pipeline.fit(X_iris_train,y_iris_train)
print(mlp_clf_pipeline.score(X_iris_test,y_iris_test))



# Implementing MLPs with Keras

# Building an Image Classifier Using the Sequential API
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# get shape and datatype
print(X_train.shape)
print(X_train.dtype, "\n")

X_train, X_valid, X_test = X_train/255, X_valid/255, X_test/255

# class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# first iamge
print(class_names[y_train[0]])



# Creating the model using the sequential API
tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28,28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# alternative way to specify layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# # plot image of model
# plot_model(model)


model.summary()
print(model.layers)

# get hidden layers
hidden1 = model.layers[1]
print(hidden1.name)
is_hidden = (model.get_layer("dense_3") is hidden1)
print(is_hidden)

# get parameter weights, biases of a layer
weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)
print(biases)
print(biases.shape, "\n")


# Compiling the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# Training and evaluating the model
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


# get learning curves
pd.DataFrame(history.history).plot(
    figsize=(8,5),
    xlim=[0,29],
    ylim=[0,1],
    grid=True,
    xlabel="Epoch",
    style=["r--", "r--", "b-", "b-*"]
)
plt.savefig("figures/learning_curve.png")


# evaluate generalization error
print(model.evaluate(X_test, y_test), "\n")


# make prediction
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2), "\n")

# get highest probable group
y_pred = y_proba.argmax(axis=-1)
print(y_pred)
print(np.array(class_names)[y_pred], "\n")  # get appropriate clothing items

# verify predictions
y_new = y_test[:3]
print(y_new, "\n")



# Building a Regression MLP Using the Sequential API

tf.random.set_seed(42)

# normalize training data
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])

# create model
model = tf.keras.Sequential([
    norm_layer,  # normalized layer
    tf.keras.layers.Dense(50, activation="relu"),  # 3 hidden layers with 50 neurons
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)  # output layer
])

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# compile model
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

# apply normalization
norm_layer.adapt(X_train)

# fit model to training data
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid,y_valid))

# evaluate model
mse_test, rmse_test = model.evaluate(X_test, y_test)

# make predictions
X_new = X_test[:3]
y_pred = model.predict(X_new)



# Building Complex Models Using the Functional API

# build layers
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30,activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30,activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

# input layer
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])

# normalize input
normalized = normalization_layer(input_)

# pass through hidden layers
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)

# concatenated layer combining wide path and deep path
concat = concat_layer([normalized, hidden2])

# ouput layer
output = output_layer(concat)

# construct model
model = tf.keras.Model(inputs=[input_], outputs=[output])



# split input inside model

# two inputs
input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7

# two normalizations
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()

# normalized inputs
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)

# apply hidden layers to deep input
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)

# concatenate wide input and deep hidden output
concat = tf.keras.layers.concatenate([norm_wide, hidden2])

# output layer
output = tf.keras.layers.Dense(1)(concat)

# make model
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])


# run split model

# get optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# compile model
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

# split training, validation, test, and new into wide and deep
X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

# # normalize input
# norm_layer_wide.adapt(X_train_wide)
# norm_layer_deep.adapt(X_train_deep)

# # fit model
# history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
#                     validation_data=((X_valid_wide, X_valid_deep), y_valid))

# # evaluate model
# mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
# print(mse_test)

# # make prediction
# y_pred = model.predict((X_new_wide, X_new_deep))
# print(y_pred, "\n")



# # add auxiliary output to split model

# # two inputs
# input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
# input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7

# # two normalizations
# norm_layer_wide = tf.keras.layers.Normalization()
# norm_layer_deep = tf.keras.layers.Normalization()

# # normalized inputs
# norm_wide = norm_layer_wide(input_wide)
# norm_deep = norm_layer_deep(input_deep)

# # apply hidden layers to deep input
# hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
# hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)

# # concatenate wide input and deep hidden output
# concat = tf.keras.layers.concatenate([norm_wide, hidden2])
# output = tf.keras.layers.Dense(1)(concat)

# # auxiliary model
# aux_output = tf.keras.layers.Dense(1)(hidden2)
# model = tf.keras.Model(inputs=[input_wide, input_deep],
#                        outputs=[output, aux_output])

# # get optimizer
# optimizer = tf.keras.optimizers.Adam(learning_rate1e-3)

# # compile model
# model.compile(loss=("mse", "mse"),
#               loss_weights=(0.9,0.1),  # set weights on loss funtions
#               optimizer=optimizer,
#               metrics=["RootMeanSquaredError"])

# # normalize input
# norm_layer_wide.adapt(X_train_wide)
# norm_layer_deep.adapt(X_train_deep)

# # fit model with two outputs (hence two labels)
# history = model.fit(
#     (X_train_wide, X_train_deep), (y_train, y_train), epochs=20,
#     validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid))
# )

# # evaluate model
# eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test,y_test))

# # get results
# weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
# print(f"weighted sum: {weighted_sum_of_losses}")
# print(f"main loss: {main_loss}")
# print(f"aux loss: {aux_loss}")
# print(f"main rmse: {main_rmse}")
# print(f"aux rmse: {aux_rmse}")

# # get predictions
# y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

# # can create dictionary output
# y_pred_tuple = model.predict((X_new_wide, X_new_deep))
# y_pred = dict(zip(model.output_names, y_pred_tuple))



# # Using the Subclassing API to Build Dynamic Models

class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # needed to support naming the model
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output
    
model = WideAndDeepModel(30, activation="relu", name="my_cool_model")



# Saving and Restoring a Model

# model.save("my_keras_model", save_format="tf")


# Load model
# model = tf.keras.model.load_model("my_keras_model")

# y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))



# Using Callbacks

# save checkpoints of model at regular intervals during training at the end of each epoch
# checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints",
#                                                    save_weights_only=True)
# history = model.fit([...], callbacks=[checkpoint_cb])



# early stopping callback combined with checkpoints
# early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
#                                                      restore_best_weights=True)
# history = model.fit([...], callbacks=[checkpoint_cb, early_stopping_cb])



# custom callback

# display ratio between validation loss and training loss during training
class PintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")



# Using TensorBoard for Visualization

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()  # e.g. my_logs/run_2022_08_01_17_25_59

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,
                                                profile_batch = (100, 200))

# history = model.fit([...], callbacks=[tensorboard_cb])



# Fine-Tuning Neural Network Hyperparameters

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

# basic random search
random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)

random_search_tuner.search(X_train, y_train, epochs=10,
                           validation_data=(X_valid, y_valid))


# get best model
top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]

# get best hyperparameters
top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
print(top3_params[0].values, "\n")  # best hyperparameter values


# ask oracle to give best trial
best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
print(best_trial.summary(), "\n")

# get metrics directly
print(best_trial.metrics.get_last_value("val_accuracy"), "\n")


# fit best model to full training set for a few epochs
best_model.fit(X_train_full, y_train_full, epochs=10)

# get test accuracy
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(test_loss)
print(test_accuracy, "\n")


class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)
    
    def fit(self, hp, model, X, y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = tf.keras.layers.Normalization()
            X = norm_layer(X)
        return model.fit(X, y, **kwargs)
    

hyperband_tuner = kt.Hyperband(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_epochs=10, factor=3, hyperband_iterations=2,
    overwrite=True, directory="my_fashion_mnist", project_name="hyperband")



root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(X_train, y_train, epochs=10,
                       validation_data=(X_valid, y_valid),
                       callbacks=[early_stopping_cb, tensorboard_cb])


# bayesian optimizer
bayesian_opt_tuner = kt.BaesianOptimization(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_trials=10, alpha=1e-4, beta=2.6,
    overwrite=True, directory="my_fashion_mnist", project_name="bayesian_opt")

# bayesian_opt_tuner.search([...])



























