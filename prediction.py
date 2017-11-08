
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from LeNet import LeNet


# Read images
my_images = []
y_labels = []
images = glob.glob('testIm/*.png')
for img in images:
    image = mpimg.imread(img)
    my_images.append(image)
    y_labels.append( int(img.split('_')[1].split('.')[0]) )
TSingTest = np.asarray(my_images)

# variable inizilitation
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
logits = LeNet(x)
saver = tf.train.Saver()
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)

#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    OutSoftmax_logits = sess.run(softmax_logits, feed_dict={x: TSingTest})
    OutTop_k = sess.run(top_k, feed_dict={x: TSingTest})
    print('SOFT MAX LOGITS: ')
    print(OutSoftmax_logits)
    print('Top K Probabilities: ')
    print(OutTop_k[0])
    print('Top K Classes: ')
    print(OutTop_k[1])
