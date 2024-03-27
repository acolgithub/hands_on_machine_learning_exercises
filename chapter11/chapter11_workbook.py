import tensorflow as tf

import math
from functools import partial
import numpy as np



# The Vanishing/Exploding Gradients Problems

# Glorot and He Initialization

dense = tf.keras.layers.Dense(50,
                              activation="relu",
                              kernel_initializer="he_normal")  # use he initializer


# make initilizer
he_avg_init = tf.keras.initializers.VarianceScaling(scale=2.,
                                                    mode="fan_avg",  # use fan_avg
                                                    distribution="uniform")  # use uniform distribution
dense = tf.keras.layers.Dense(50,
                               activation="sigmoid",
                               kernel_initializer=he_avg_init)



# Better Activation Functions

# Leaky ReLU
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)  # defaults to alpha=0.3
dense = tf.keras.layers.Dense(50,
                              activation=leaky_relu,
                              kernel_initializer="he_normal")

# can also use LeakyReLU as a separate layer in model
# model = tf.keras.models.Sequential([
# [...]  # more layers
# tf.keras.layers.Dense(50, kernel_initializer="he_normal")  # np activation
# tf.keras.layers.LeakyReLU(alpha=0.2),  # activation as a separate layer
# [...]  # more layers
# ])


# similar things can be done with PReLU, ELU,SELU



# Implementing batch normalization with Kreas
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

print([(var.name, var.trainable) for var in model.layers[1].variables])


# batch normalization after  layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),  # remove previous layer bias term
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),  # add activation layer instead of setting it inside dnese layer
    tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])


# Gradient Clipping
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
optimizer = tf.keras.optimizers.SGD(clipnorm=1.0)


# Transfer Learning with Keras
# [...]  # Assuming model A was already trained and saved to my "my_model_A"
# model_A = tf.keras.models.load_model("my_model_A")  # load model A
# model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])  # reuse all but last layer
# model_B_on_A.add(tf.keras.layers.Dense(1,activation="sigmoid"))  # add new final layer


# clone model to avoid affecting model A
# model_A_clone = tf.keras.models.clone_model(model_A)
# model_A_clone.set_weights(model_A.get_weights())


# freeze the reused layers during the first few epochs
# for layer in model_B_on_A.layers[:-1]:
#   layer.trainable = False
#
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
# model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
#           metrics=["accuracy"])  # need to compile model after you freeze or unfreeze layers
#
# history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
#                 validation_data=(X_valid_B, y_valid_B))
#
# for layer in model_B_on_A.layers[:-1]:
#   layer.trainable = True
# 
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
# model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
#           metrics=["accuracy"])
# history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
#                 validation_data=(X_valid_B, y_valid_B))



# Faster Optimizers

# Momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)



# Nesterov Accelerated Gradient
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,
                                    nesterov=True)



# RMSProp
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)



# Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                     beta_2=0.999)




# Learning Rate Scheduling

# power scheduling
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, weight_decay=1e-4)



# exponential scheduling
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch/s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)



# learning rate scheduler
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])



# piecewise constant scheduling
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
    


# reduce LR on plateau
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])



# alternative implementation of learning rate scheduling
# batch_size = 32
# n_epochs = 25
# n_steps = n_epochs * math.ceil(len(X_train) / batch_size)
# scheduled_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1)
# optimizer = tf.keras.optimizers.SGD(learning_rate=scheduled_learning_rate)



# Avoiding Overfitting Through Regularization

# l1 and l2 Regularization
layer = tf.keras.layers.Dense(100, activation="relu",
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))

layer = tf.keras.layers.Dense(100, activation="relu",
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l1(0.01))

layer = tf.keras.layers.Dense(100, activation="relu",
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.01))



RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    RegularizedDense(100),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])



# Dropout
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])




# Montre Carlo (MC) Dropout
# y_probas = np.stack([model(X_test, training=True)
                    #  for sample in range(100)])
# y_proba = y_probas.mean(axis=0)



class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)










