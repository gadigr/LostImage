import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from os import listdir
from os.path import isfile, join
from PIL import Image
from resizeimage import resizeimage
from skimage import img_as_float

size = 28,28
def PIL2array(img):
    ar = np.array(img.getdata(),
                    np.float32)
    return np.multiply(ar, 1/255).tolist()

# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# some , other = mnist.train.next_batch(50)
# print(other)
def readData():
    dirs = listdir('letters/')
    y = []
    x =[]
    for directory in dirs:
        images = listdir('letters/'+directory)

        for img in images:
            arr = [0.] * 50
            arr[int(directory) - 1] = 1.
            y.append(arr)
            fd_img = 'letters/'+directory + '/' + img
            imgg = Image.open(fd_img).convert('L')
            imgg = imgg.resize(size)
            x.append(PIL2array(imgg))
    
    return x,y

# print(readData())

# Convolution definition
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

# functions for parameter initialization 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# We start
n_input = 28*28
n_output = 50

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

x_tensor = tf.reshape(x, [-1, 28, 28, 1])

# Weight matrix is [height x width x input_channels x output_channels]
# Bias is [output_channels]
filter_size = 5
n_filters_1 = 16
n_filters_2 = 16

# parameters
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])

# layers
h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 7x7 is the size of the image after the convolutional and pooling layers (28x28 -> 14x14 -> 7x7)
h_conv2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * n_filters_2])

# %% Create the one fully-connected layer:
n_fc = 1024
W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([n_fc, n_output])
b_fc2 = bias_variable([n_output])
y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# We'll train in minibatches and report accuracy:
batch_size = 100
n_epochs = 5
l_loss = list()

images,labels = readData()
for epoch_i in range(n_epochs):
    for batch_i in range(0, 30):
        batch_xs, batch_ys = readData()
        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})
    loss = sess.run(accuracy, feed_dict={
                        x: images,
                        y: labels,
                        keep_prob: 1.0 })
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    l_loss.append(loss)


print("Accuracy for test set: {}".format(sess.run(accuracy,
                feed_dict={
                    x: images,
                    y: labels,
                    keep_prob: 1.0
                })))