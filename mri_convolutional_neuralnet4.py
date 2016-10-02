import os
import math
import numpy as np
import tensorflow as tf
from utils import *
from scipy.ndimage.interpolation import zoom

i_max = 1480
train_x, train_y = load_train_data()

#train_y = 48~97, label = (y - 48)/5

min_age, max_age = min(train_y), max(train_y)

dn = 1.4
d0, d1, d2 = 360, 300, 216 #360, 512, 216
d1_origin = 512
d0, d1, d2 = round(d0/dn), round(d1/dn), round(d2/dn)
d = d0 * d1 * d2

n_output = 1 
p = 3 # stride size in pooling layer

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, p, p, p, 1], 
                        strides=[1, p, p, p, 1], padding='SAME')


def restore(saver, sess, name=''):
  fname = "./tmp/model_" + name + ".ckpt"
  if os.path.isfile(fname):
    saver.restore(sess, fname)

 
with tf.device('/cpu:0'):
  x = tf.placeholder(tf.float32, shape=[None, d])
  y_ = tf.placeholder(tf.float32, shape=[None, n_output])
  W_conv1 = weight_variable([3, 3, 3, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1, d0, d1, d2, 1])
  h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2x2(h_conv1)

  W_conv2 = weight_variable([3, 3, 3, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2x2(h_conv2)
 
  W_conv3 = weight_variable([3, 3, 3, 64, 128])
  b_conv3 = bias_variable([128])
  h_conv3 = tf.nn.relu(conv3d(h_pool2, W_conv3) + b_conv3)
  h_pool3 = max_pool_2x2x2(h_conv3)
 
  n_pool = 3
  dt = p ** n_pool
  drow = math.ceil(d0/dt) * math.ceil(d1/dt) * math.ceil(d2/dt) * 128 # drow = 659456, vulnerable 
  h_pool3_flat = tf.reshape(h_pool3, [-1, drow])
 
  W_fc1 = weight_variable([drow, 1024])
  b_fc1 = bias_variable([1024])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
  
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  W_fc3 = weight_variable([10, 1])
  b_fc3 = bias_variable([1])
  y_conv= tf.matmul(h_fc2_drop, W_fc3) + b_fc3
 
  error = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_, y_conv))))

with tf.device('/gpu:0'):
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(error)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

saver_conv1 = tf.train.Saver({"w": W_conv1, "b": b_conv1})
saver_conv2 = tf.train.Saver({"w": W_conv2, "b": b_conv2})
saver_conv3 = tf.train.Saver({"w": W_conv3, "b": b_conv3})
saver_fc = tf.train.Saver({"w1": W_fc1, "b1": b_fc1, "w2": W_fc2, "b2": b_fc2, "w3": W_fc3, "b3": b_fc3})

restore(saver_conv1, sess, name='conv1')
restore(saver_conv2, sess, name='conv2')
restore(saver_conv3, sess, name='conv3')
restore(saver_fc, sess, name='fc')

for i in range(20000):
  err = 0.0
  accu = 0.0
  for j in range(len(train_x)):
    print(i,j)
    shape = train_x[j].shape
    if round(shape[0]/dn) != d0 or shape[1] != d1_origin or round(shape[2]/dn) != d2:  
      continue #14
    batch_x = train_x[j].get_data()
    batch_x = batch_x[:,106:406,:]
    batch_x = zoom(batch_x, 1/dn)
    print(batch_x.shape)
    batch_x = (batch_x.reshape(1, d) - 53) / i_max
    batch_y = np.array([[train_y[j]]])
    fetches = [train_step, error, y_conv]
    t = sess.run(fetches, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.1})
    err += t[1] 
    pred = t[2] 
    print(i, j, err, pred)

  saver_conv1.save(sess, "./tmp/model_conv1.ckpt")
  saver_conv2.save(sess, "./tmp/model_conv2.ckpt")
  saver_conv3.save(sess, "./tmp/model_conv3.ckpt")
  saver_fc.save(sess, "./tmp/model_fc.ckpt")
