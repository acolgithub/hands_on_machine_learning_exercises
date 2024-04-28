



# Q1: What is the problem that Glorot initialization and He initialization aim to fix?

print(f"""Glorot initialization and He initialization aim to fix the problem of
vanishing and exploding gradients.\n""")



# Q2: Is it OK to initialize all weights to the same value as long as that
#     value is selected randomly using He initialization?

print(f"""No, it is important to try and initialize the weights in
an independent manner. If they are initialized to be eqyal backpropagation
will not be able to make them unequal.\n""")



# Q3: It is OK to initialize the bias terms to 0?

print(f"""Yes, it is okay to initialize the first bias term to be zero.\n""")



# Q4: In which cases would you want to use each of the activation functions
#     we discussed in this chapter?

print(f"""We would use Leaky ReLU or Patametric Leaky ReLU if we run into
problems involving dying neurons. It could also be used as a performance
boost over ReLU. We could also try an Exponential Linear Unit or a
Scaled Exponential Linear Unit however this could result in a slower
computation. For further performance improvements, as well as for handling
complex tasks, we could make use of GELU, Swish and Mish.\n""")



# Q5: What may happen if you set the momentum hyperparameter too close to
#     1 (e.g., 0.99999) when using an SGD optimizer?

print(f"""If the momentum parameter is set too close to 1 then SGD may
consistently overshoot the minimizer or converge very slowly. Note that
for a constant gradient setting the momentum parameter close to 1 puts
no upper bound on terminal speed.\n""")



# Q6: Name three ways you can produce a sparse model.

print(f"""You can produce a sparse model by training as usual and getting
rid of the tiny weights by setting them to zero, applying strong l1 regularization
during training, or using the TensorFlow Model Optimization Toolkit.\n""")



# Q7: Does dropout slow down training? Does it slow down inference (i.e.,
#     making predictions on new instances)? What about MC dropout?

print(f"""Dropout slows down convergence and hence slows down training.
Dropout does not slow down predictions and actually may result in a
better model. MC Dropout will also slow convergence since you will want
to train multiple models and average them.\n""")



# Q8: Practice training a deep neural network on the CIFAR10 image dataset:
#
#     a. Build a DNN with 20 hidden layers of 100 neurons each (that's too
#        many, but it's the point of this exercise). Use He initialization and
#        the Swish activation function.
#     b. Using Nadam optimization and early stopping, train the network on
#        the CIFAR10 dataset. You can load it with
#        tf.keras.datasets.cifar10.load_data(). The dataset is composed of
#        60,000 32 x 32-pixel color images (50,000 for training, 10,000 for
#        testing) with 10 classes, so you'll need a softmax output layer with
#        10 neurons. Remember to search for the right learning rate each
#        time you change the mode's architecture or hyperparameters.
#     c. Now try adding batch normalization and compare the learning
#        curves: is it converging faster than before? Does it produce a better
#        model? How does it affect training speed?
#     d. Try replacing batch normalization with SELU, and make the
#        necessary adjustments to ensure the network self-normalizes (i.e.,
#        standardize the input features, use LeCun normal initialization,
#        make sure the DNN contains only a sequence of dense layers, etc.).

import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data

# data
data = load_data()

# get training and testing sets
(X_train_total, y_train_total), (X_test, y_test) = data

# get validation data
X_valid, y_valid = X_train_total[:5000], y_train_total[:5000]
X_train, y_train = X_train_total[5000:], y_train_total[5000:]

# a., b., c., d.

# make model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3]))  # flatten input

# add 20 hidden layers of 100 neurons
for i in range(20):
    model.add(tf.keras.layers.Dense(100, kernel_initializer="lecun_normal", activation="selu"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Activation("swish"))
    
# add output layer
model.add(tf.keras.layers.AlphaDropout(rate=0.2))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# get optimizer
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-5)

# compile model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

# checkpoint
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints",
                                                   save_weights_only=True)

# early stopping
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)

# train model
model.fit(X_train,
          y_train,
          epochs=100,
          validation_data=(X_valid, y_valid),
          callbacks=[checkpoint_cb, early_stopping_cb])



