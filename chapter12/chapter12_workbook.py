import tensorflow as tf

import numpy as np



# Tensors and Operations
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])  # matrix
print(t)

# get shape and type
print(t.shape)
print(t.dtype, "\n")


# print first and second column
print(t[:, 1:], "\n")

# add constant to tensor
print(t+10)
print(tf.add(t, 10), "\n")  # equivalent

# square entries of tensor
print(tf.square(t), "\n")

# compute product of tensor and its transpose
print(t@tf.transpose(t))
print(tf.matmul(t, tf.transpose(t)), "\n")

# scalar tensor
print(tf.constant(42), "\n")




# Tensors and NumPy
a = np.array([2., 4., 5.])
print(tf.constant(a), "\n")

# convert to numpy array
print(t.numpy())
print(np.array(t), "\n")

# square tensor
print(tf.square(a))
print(np.square(t), "\n")



# Type Conversions
t2 = tf.constant(40, dtype=tf.float64)
print(tf.constant(2.0) + tf.cast(t2, tf.float32), "\n")



# Variables





























