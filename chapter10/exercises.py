



# Q3: Why is it generally preferable to use a logistic regression classifier
#     rather than a classic perceptron (i.e., a single layer of threshold logic
#     units trained using the perceptron training algorithm)? How can you
#     tweak a perceptron to make it equivaent to a logistic regression
#     classifier?

print(f"""A single layer of TLUs only provides a binary classification
based on inputs while a logistic regression classifier gives a probability
indicating certainty of which classification it gives.
A single layer using a sigmoid activation function would
make a single layer of TLUs equivalent to a logistic regression classifier.\n""")



# Q4: Why was the sigmoid activation function a key ingredient in training the
#     first MLPs?

print(f"""The sigmoid function was important in training the first MLPs
since it has well-defined non-zero derivative everywhere which allows
gradient descent to make progress each step.\n""")



# Q5: Name three popular activation functions. Can you draw them?

print(f"""Three popular activation functions are sigmoid, hyperbolic tangent,
and the rectified linear unit.\n""")



# Q6: Suppose you have an MLP compound of one input layer with 10
#     passthrough neurons, followed by one hidden layer with 50 artificial
#     neurons, and finally one output layer with 3 artificial neurons. All
#     artificial neurons use the ReLU activation function.
#
#     a. What is the shape of the input matrix X?
#     b. What are the shapes of the hidden layer's weight matrix Wh and
#        bias vector bh?
#     c. What are the shapes of the output layer's weight matrix Wo and
#        bias vector bo?
#     d. What is the shape of the network's output matrix Y?
#     e. Write the equation that computes the network's output matrix Y as
#        a function of X, Wh, bh, Wo, and bo.

# a.
print(f"""The input matrix, X, has shape mx10 where m is the number of
inputs.\n""")

# b.
print(f"""The hidden layer's weight matrix has shape 10x50.
The bias vector has shape 1x50.\n""")

# c.
print(f"""The output layer's weight matrix has shape 50x3.
The bias vector has shape 1x3.\n""")

# d.
print(f"""The shape of the output matrix is mx3 where m is the number of
inputs.\n""")

# e.
print(f"""Y = ReLU(ReLu(XW_h + b_h)W_o + b_o)\n""")



# Q8: What is backpropagation and how does it work? What is the difference
#     between backpropagation and reverse-mode autodiff?

print(f"""Backpropagation is an algorithm to calculate the amount of
difference caused in the ouput from its target due to each of its
layers using chain rule and proceeding backward through the neural
network.
This is used to help train a neural network by applying a gradient
descent to improve the accuracy.
Reverse-mode autodiff refers to both the forward and backward passes
while backpropagation refers only to the backward pass.\n""")



# Q9: Can you list all the hyperparameters you can tweak in a basic MLP? If
#     the MLP overfits the training data, how could you tweak these
#     hyperparameters to try to solve the problem?

print(f"""Hyperparameters that can be tweaked include the number of
hidden layers, the number of neurons per hidden layer,
the number of output neurons, the activation functions, and the loss function.\n""")



# Q10: Train a deep MLP on the MNIST dataset (you can load it using tf.keras.
#      dataset.mnist.load_data()). See if you can get over 98% accuracy by
#      manually tuning the hyperparameters. Try searching for the optimal
#      learning rate by using the approach presented in this chapter (i.e., by
#      growing the learning rate exponentially, plotting the loss, and finding
#      the point where the loss shoots up). Next, try tuning the hyperparameters
#      using Keras Tuner with all the bells and whistles--save checkpoints, use
#      early stopping, and plot learning curves using TensorBoard.

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data

import matplotlib.pyplot as plt
import pandas as pd

# data
data = load_data()

# get train and test sets
X_train_start, y_train_start= data[0][0]/255, data[0][1]  # divided by 255 to obtain floats in 0-1 range for standardizing
X_test, y_test = data[1][0]/255, data[1][1]

# get validation set
X_train, y_train = X_train_start[5000:], y_train_start[5000:]
X_val, y_val = X_train_start[:5000], y_train_start[:5000]

# construct model
tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28,28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="sigmoid"))
model.add(tf.keras.layers.Dense(100, activation="sigmoid"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# get sgd
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=1e0)

# compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=sgd_optimizer,
    metrics=["accuracy"]
)

# train model
history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    validation_data=(X_val,y_val))

# plot training
pd.DataFrame(history.history).plot(
    figsize=(8,5),
    xlim=[0,29],
    ylim=[0,1],
    grid=True,
    xlabel="Epoch",
    style=["r--", "r--", "b-", "b-*"]
)
plt.show()
model.evaluate(X_test, y_test)






















