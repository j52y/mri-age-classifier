import os
import math
import numpy as np
import tensorflow as tf
from utils import *
from scipy.ndimage.interpolation import zoom

train_x, train_y = load_train_data()
min_age, max_age = min(train_y), max(train_y)

dn = 2
o0, o1, o2 = 300, 300, 200 #360, 512, 216
d0, d1, d2 = round(o0/dn), round(o1/dn), round(o2/dn)
d = d0 * d1 * d2

n_output = 1 
p = 2 # stride size in pooling layer


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

  x_image = tf.reshape(x, [-1, d0, d1, d2, 1])

  W_conv1a = weight_variable([3, 3, 3, 1, 64])
  b_conv1a = bias_variable([64])
  h_conv1a = tf.nn.relu(conv3d(x_image, W_conv1a) + b_conv1a)

  W_conv1b = weight_variable([3, 3, 3, 64, 64])
  b_conv1b = bias_variable([64])
  x_image = tf.reshape(x, [-1, d0, d1, d2, 1])
  h_conv1b = tf.nn.relu(conv3d(h_conv1a, W_conv1b) + b_conv1b)

  h_pool1 = max_pool_2x2x2(h_conv1b)

  W_conv2a = weight_variable([3, 3, 3, 64, 128])
  b_conv2a = bias_variable([128])
  h_conv2a = tf.nn.relu(conv3d(h_pool1, W_conv2a) + b_conv2a)

  W_conv2b = weight_variable([3, 3, 3, 128, 128])
  b_conv2b = bias_variable([128])
  h_conv2b = tf.nn.relu(conv3d(h_conv2a, W_conv2b) + b_conv2b)

  h_pool2 = max_pool_2x2x2(h_conv2b)
 
  W_conv3a = weight_variable([3, 3, 3, 128, 256])
  b_conv3a = bias_variable([256])
  h_conv3a = tf.nn.relu(conv3d(h_pool2, W_conv3a) + b_conv3a)
 
  W_conv3b = weight_variable([3, 3, 3, 256, 256])
  b_conv3b = bias_variable([256])
  h_conv3b = tf.nn.relu(conv3d(h_conv3a, W_conv3b) + b_conv3b)

  h_pool3 = max_pool_2x2x2(h_conv3b)

  W_conv4a = weight_variable([3, 3, 3, 256, 512])
  b_conv4a = bias_variable([512])
  h_conv4a = tf.nn.relu(conv3d(h_pool3, W_conv4a) + b_conv4a)
 
  W_conv4b = weight_variable([3, 3, 3, 512, 512])
  b_conv4b = bias_variable([512])
  h_conv4b = tf.nn.relu(conv3d(h_conv4a, W_conv4b) + b_conv4b)
 
  W_conv4c = weight_variable([3, 3, 3, 512, 512])
  b_conv4c = bias_variable([512])
  h_conv4c = tf.nn.relu(conv3d(h_conv4b, W_conv4c) + b_conv4c)

  h_pool4 = max_pool_2x2x2(h_conv4c)

  W_conv5a = weight_variable([3, 3, 3, 512, 512])
  b_conv5a = bias_variable([512])
  h_conv5a = tf.nn.relu(conv3d(h_pool4, W_conv5a) + b_conv5a)
 
  W_conv5b = weight_variable([3, 3, 3, 512, 512])
  b_conv5b = bias_variable([512])
  h_conv5b = tf.nn.relu(conv3d(h_conv5a, W_conv5b) + b_conv5b)
 
  W_conv5c = weight_variable([3, 3, 3, 512, 512])
  b_conv5c = bias_variable([512])
  h_conv5c = tf.nn.relu(conv3d(h_conv5b, W_conv5c) + b_conv5c)

  h_pool5 = max_pool_2x2x2(h_conv5c)

  n_pool = 5
  dt = 2 ** n_pool
  drow = math.ceil(d0/dt) * math.ceil(d1/dt) * math.ceil(d2/dt) * 512 #  vulnerable 
  h_pool5_flat = tf.reshape(h_pool5, [-1, drow])
  
  keep_prob = tf.placeholder(tf.float32)

with tf.device('/gpu:0'):
  W_fc0 = weight_variable([drow, 8192])
  b_fc0 = bias_variable([8192])
  h_fc0 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc0) + b_fc0)
  h_fc0_drop = tf.nn.dropout(h_fc0, keep_prob)

  W_fc1 = weight_variable([8192, 1024])
  b_fc1 = bias_variable([1024])
  h_fc1 = tf.nn.relu(tf.matmul(h_fc0_drop, W_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  W_fc3 = weight_variable([10, 1])
  b_fc3 = bias_variable([1])
  y_conv= tf.matmul(h_fc2_drop, W_fc3) + b_fc3
 
  #error = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_, y_conv))))
  error = tf.abs(tf.sub(y_, y_conv))

train_step = tf.train.GradientDescentOptimizer(2e-5).minimize(error)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
restore(saver, sess, name='3dim_vgg')


for i in range(20000):
  error_sum = 0.0
  for j in range(len(train_x)):
    err = 0.0
    shape = train_x[j].shape
    batch_x = normalize_image(crop_image(train_x[j].get_data(), [o0, o1, o2]))
    r0, r1, r2 = np.random.choice(o0, d0), np.random.choice(o1, d1), np.random.choice(o2, d2)
    batch_x = batch_x[r0,:,:][:,r1,:][:,:,r2].reshape(1, d)
    batch_y = np.array([[train_y[j]]])
    fetches = [train_step, error, y_conv]
    t = sess.run(fetches, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.3})
    err = t[1]
    error_sum += err[0][0]
    pred = t[2]
    print(i, j, train_y[j], pred[0][0], err, error_sum)
    if j%2 == 0:
      msg = '{} {} {} {} {}'.format(i, j, pred[0][0], err[0][0], error_sum)
      os.system("curl \"https://api.telegram.org/bot236245101:AAFZ12aHX2emHeZuU99R11TdWMk9fZfl1j0/sendMessage?chat_id=237652977&text=" + msg + "\"")
      print('')
    saver.save(sess, "./tmp/model_3dim_vgg.ckpt")
