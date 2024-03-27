



# 1.
print(f"""Glorot initialization and He initialization aim to fix the problem of
vanishing and exploding gradients.\n""")



# 2.
print(f"""No, it is important to try and initialize the weights in
an independent manner. If they are initialized to be eqyal backpropagation
will not be able to make them unequal.\n""")



# 3.
print(f"""Yes, it is okay to initialize the first bias term to be zero.\n""")



# 4.
print(f"""We would use Leaky ReLU or Patametric Leaky ReLU if we run into
problems involving dying neurons. It could also be used as a performance
boost over ReLU. We could also try an Exponential Linear Unit or a
Scaled Exponential Linear Unit however this could result in a slower
computation. For further performance improvements, as well as for handling
complex tasks, we could make use of GELU, Swish and Mish.\n""")



# 5.
print(f"""If the momentum parameter is set too close to 1 then SGD may
consistently overshoot the minimizer or converge very slowly. Note that
for a constant gradient setting the momentum parameter close to 1 puts
no upper bound on terminal speed.\n""")



# 6.
print(f"""You can produce a sparse model by training as usual and getting
rid of the tiny weights by setting them to zero, applying strong l1 regularization
during training, or using the TensorFlow Model Optimization Toolkit.\n""")



# 7.
print(f"""Dropout slows down convergence and hence slows down training.
Dropout does not slow down predictions and actually may result in a
better model. MC Dropout will also slow convergence since you will want
to train multiple models and average them.\n""")



# 8.
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






















