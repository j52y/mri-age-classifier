import numpy as np
import tensorflow as tf
from utils import *
from scipy.ndimage.interpolation import zoom

# r_range = 0.1
i_max = 1480
train_x, train_y = load_train_data()

min_age, max_age = min(train_y), max(train_y)

original_row, original_col = 360, 512
original_size = original_row * original_col
n_row, n_col = 28, 28
n_input = n_row * n_col 
n_output = 10 # len(set(train_y))

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, n_input])
y_ = tf.placeholder(tf.float32, shape=[None, n_output])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,n_row,n_col,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  accu = 0.0
  for j in range(len(train_x)):
    if train_x[j].shape[0] * train_x[j].shape[1] != original_size:
      continue
    batch_x = np.max(train_x[j].get_data(), axis=2)
    batch_x = batch_x[40:320, 116:396]
    batch_x = zoom(batch_x, 0.1)
    batch_x = batch_x.reshape(1, n_input) / i_max
    # batch_y = np.array([[train_y[j]/(max_age - min_age)]]) 
    # batch_y = (np.arange(10)/10).reshape(-1,10)
    batch_y = np.zeros(10)
    batch_y[int(train_y[j]/10)] = 1
    print(train_y[j], batch_y)
    batch_y = batch_y.reshape(-1,10)
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    accu = y_conv.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    print(i, j, accu)
