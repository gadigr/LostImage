import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from os import listdir
import os.path
from os.path import isfile, join
from PIL import Image
from resizeimage import resizeimage
from skimage import img_as_float
import pickle
import random
import datetime
import math

lr = 0.0007 # Set learning rate
batch_size = 100 # Set batch size
n_epochs = 1700 # Set number of epochs
sizing = 30
size = sizing,sizing # Set data size
test_to_data_ratio = 0.85 # Set train to test ration
def PIL2array(img):
    ar = np.array(img.getdata(),
                    np.float32)
    return np.multiply(ar, 1/255).tolist()

# reads data from files
def readData():
    dirs = listdir('FinalDataset/')
    y = []
    x =[]
    for directory in dirs:
        images = listdir('FinalDataset/'+directory)

        for img in images:
            arr = [0.] * 50
            arr[int(directory) - 1] = 1.
            y.append(arr)
            fd_img = 'FinalDataset/'+directory + '/' + img
            imgg = Image.open(fd_img).convert('L')
            imgg = imgg.resize(size)
            x.append(PIL2array(imgg))
    
    return x,y

def shuffle(x,y):
    zipped = list(zip(x,y))
    random.shuffle(zipped)
    n_x,n_y = zip(*zipped)
    return n_x, n_y
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
    initial = tf.constant(0.1, shape=shape) # Sets lr
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)


# We start
n_input = sizing*sizing
n_output = 50
sizeAfterConv = int(math.ceil(sizing/4))
with tf.name_scope('input_layer'):
    x = tf.placeholder(tf.float32, [None, n_input])
    x_tensor = tf.reshape(x, [-1, sizing, sizing, 1])

# Weight matrix is [height x width x input_channels x output_channels]
# Bias is [output_channels]
filter_size = 5
n_filters_1 = 20
n_filters_2 = 20

# parameters and layers
with tf.name_scope('First_Convolution'):
    W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])
    b_conv1 = bias_variable([n_filters_1])
    with tf.name_scope('activation'):
        h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1) + b_conv1)
    with tf.name_scope('max_pool'):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('Second_Convolution'):
    W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
    b_conv2 = bias_variable([n_filters_2])
    with tf.name_scope('activation'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('max_pool'):
        h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('flatten'):
    # 8x8 is the size of the image after the convolutional and pooling layers (30x30 -> 15x15 -> 8x8)
    h_conv2_flat = tf.reshape(h_pool2, [-1, sizeAfterConv*sizeAfterConv * n_filters_2])

# %% Create the one fully-connected layers:
with tf.name_scope('2048_fully_connected'):
    n_fc = 2048
    W_fc1 = weight_variable([sizeAfterConv*sizeAfterConv * n_filters_2, n_fc])
    b_fc1 = bias_variable([n_fc])
    with tf.name_scope('activation'):
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

with tf.name_scope('1024_fully_connected'):
    n_fc2 = 1024
    W_fc2 = weight_variable([n_fc,n_fc2])
    b_fc2 = bias_variable([n_fc2])
    with tf.name_scope('activation'):
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2)+b_fc2)
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

with tf.name_scope('output'):
    W_fc3 = weight_variable([n_fc2, n_output])
    b_fc3 = bias_variable([n_output])
    y_pred = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    y = tf.placeholder(tf.float32, [None, n_output])
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    variable_summaries(y)
tf.summary.scalar('cross_entropy', cross_entropy)



with tf.name_scope('adam_optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

with tf.name_scope('loss_function'):
    # Creating loss function
    with tf.name_scope('prediction'):
        prediction = tf.argmax(y_pred, 1)
        need_result = tf.argmax(y, 1)
        correct_prediction = tf.equal(prediction, need_result)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float')) 
tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('confusion_matrix'):
    confusion = tf.contrib.metrics.confusion_matrix(prediction,need_result)

with tf.name_scope('init'):
    # variables:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
# init logger
date = datetime.datetime.now().timestamp()
train_writer = tf.summary.FileWriter('graphs/{}/{}/{}/{}/{}/{}/train'.format(test_to_data_ratio,lr,batch_size,n_epochs,sizing,date), sess.graph)
test_writer = tf.summary.FileWriter('graphs/{}/{}/{}/{}/{}/{}/test'.format(test_to_data_ratio,lr,batch_size,n_epochs,sizing,date))
merged = tf.summary.merge_all()

# We'll train in minibatches and report accuracy:
l_loss = list()
images = []
labels = []

# Checking for preprocessed data
imageFileName = 'images{}.dat'.format(sizing)
labelsFileName = 'labels{}.dat'.format(sizing)
if (os.path.exists(imageFileName) and os.path.exists(labelsFileName)):
    with open(imageFileName,'rb') as fp:
        images = pickle.load(fp)
    with open(labelsFileName,'rb') as fp:
        labels = pickle.load(fp)
else:
    images,labels = readData()
    with open(imageFileName.format(sizing), 'wb') as fp:
        pickle.dump(images, fp)
    with open(labelsFileName.format(sizing), 'wb') as fp:
        pickle.dump(labels, fp)

# Creating test and train data sets
train_length = int(len(images)*test_to_data_ratio)
images, labels = shuffle(images,labels) # Shuffeling
train_x = images[:train_length]
train_y = labels[:train_length]
test_x = images[train_length+1:]
test_y=labels[train_length+1:]





for epoch_i in range(n_epochs):
    batch_x = train_x[batch_size*((epoch_i+1) % 10):batch_size*(((epoch_i+1) % 10)+1)]
    batch_y = train_y[batch_size*((epoch_i+1) % 10):batch_size*(((epoch_i+1) % 10)+1)]
    summary, loss = sess.run([merged,optimizer], feed_dict={
        x: batch_x, y: batch_y, keep_prob: 0.5})

    # Running test every 10
    if ((epoch_i+1) % 10 == 0):
        train_writer.add_summary(summary,epoch_i)
        summary, loss,confuse = sess.run([merged,accuracy, confusion], feed_dict={
                        x: test_x,
                        y: test_y,
                        keep_prob: 1.0 })
        test_writer.add_summary(summary, epoch_i)
        print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
        l_loss.append(loss)
        train_x , train_y = shuffle(train_x,train_y)

summary, loss, confuse =sess.run([merged,accuracy,confusion],feed_dict={
                    x: test_x,
                    y: test_y,
                    keep_prob: 1.0})
np.savetxt('confusions/{}_{}_{}_{}_{}_{}.csv'.format(test_to_data_ratio,lr,batch_size,n_epochs,sizing,date),confuse,delimiter=',')
print("Accuracy for test set: {}".format(loss))
test_writer.close()
train_writer.close()