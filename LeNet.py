from tensorflow.contrib.layers import flatten
import tensorflow as tf


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    #sigma = 0.1
    sigma = 0.01

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x158.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 158), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(158))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x158. Output = 14x14x158.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Input 14x14x158 Output = 10x10x250.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 158, 250), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(250))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x250. Output = 5x5x250.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Dropout
    conv2 = tf.nn.dropout(conv2, 0.5)

    # Flatten. Input = 5x5x250. Output = 6250.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 6250. Output = 1000.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(6250, 1000), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(1000))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 1000. Output = 100.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(1000, 100), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(100))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    #  Layer 5: Fully Connected. Input = 100. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(100, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
