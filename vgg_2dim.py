import os
import math
import numpy as np
import tensorflow as tf
from utils import *
from scipy.ndimage.interpolation import zoom


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.03)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def fc2d(x, d0, d1, drop):
  W_fc = weight_variable([d0, d1])
  b_fc = bias_variable([d1])
  h_fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc)
  h_fc_drop = tf.nn.dropout(h_fc, drop)
  return h_fc_drop


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def convl(l, r1, r2):
  W_conv = weight_variable([3, 3, r1, r2])
  b_conv = bias_variable([r2])
  h_conv = tf.nn.relu(conv2d(l, W_conv) + b_conv)
  return h_conv


def vgg2d(x, d0, d1):
  d = d0 * d1
  x_image = tf.reshape(x, [-1, d0, d1, 1])

  conv1a = convl(x_image, 1, 64)
  conv1b = convl(conv1a, 64, 64)
  h_pool1 = max_pool_2x2(conv1b)

  conv2a = convl(h_pool1, 64, 128)
  conv2b = convl(conv2a, 128, 128)
  h_pool2 = max_pool_2x2(conv2b)

  conv3a = convl(h_pool2, 128, 256) 
  conv3b = convl(conv3a, 256, 256) 
  h_pool3 = max_pool_2x2(conv3b)

  conv4a = convl(h_pool3, 256, 512)
  conv4b = convl(conv4a, 512, 512)
  h_pool4 = max_pool_2x2(conv4b)
 
  conv5a = convl(h_pool4, 512, 1024)
  conv5b = convl(conv5a, 1024, 1024)
  conv5c = convl(conv5b, 1024, 1024)
  h_pool5 = max_pool_2x2(conv5c)

  conv6a = convl(h_pool5, 1024, 2048)
  conv6b = convl(conv6a, 2048, 2048)
  conv6c = convl(conv6b, 2048, 2048)
  h_pool6 = max_pool_2x2(conv6c)

  n_pool = 6
  dt = 2 ** n_pool
  drow = math.ceil(d0/dt) * math.ceil(d1/dt) *  2048 #  vulnerable 
  h_pool6_flat = tf.reshape(h_pool6, [-1, drow])

  return h_pool6_flat, drow


def restore(saver, sess, name=''):
  fname = "./tmp/model_" + name + ".ckpt"
  if os.path.isfile(fname):
    saver.restore(sess, fname)


younger = 25
older = 72.14
padding = 20.38


def fetch(keys):
  fetch_x = [np.load('./data/ixi_mra_mip/crop/' + str(k) + '.npy').reshape(-1) for k in keys]
  fetch_y = [[1,0,0] if ages[k] <= younger else [0,0,1] if ages[k] > older else [0,1,0] for k in keys]
  return fetch_x, fetch_y


def divide_set(keys):
  trains = [k for i, k in enumerate(keys) if i % 9 != 0]
  valids = [k for i, k in enumerate(keys) if i % 9 == 0]
  return trains, valids


imgs, ages  = load_ixi_data()
del imgs

youngs = [k for k, v in ages.items() if v <= younger] 
middles = [k for k, v in ages.items() if v >= younger + padding and v < older - padding]
olds = [k for k, v in ages.items() if v > older]

print("The Number of Samples per Class: ", len(youngs), len(middles), len(olds))

young_train, young_valid = divide_set(youngs)
middle_train, middle_valid = divide_set(middles) 
old_train, old_valid = divide_set(olds)

x_train = np.vstack((young_train, middle_train, old_train)).reshape((-1,),order='F') #interweave
x_valid = np.array(young_valid + middle_valid + old_valid)

o0, o1, o2 = 300, 450, 100 #512, 512, 100
d0, d1, d2 = o1 * o2, o0 * o2, o0 * o1 

n_output = 3 

with tf.device('/gpu:0'):
  x = tf.placeholder(tf.float32, shape=[None, d2])
  y_ = tf.placeholder(tf.float32, shape=[None, n_output])
  drop = tf.placeholder(tf.float32)

  h_vgg, r2 = vgg2d(x, o0, o1)
  h_fc0 = fc2d(h_vgg, r2, 4096, drop) 
  h_fc1 = fc2d(h_fc0, 4096, 1024, drop)
  h_fc2 = fc2d(h_fc1, 1024, 128, drop)
  h_fc3 = fc2d(h_fc2, 128, 10, drop)

  W_fc3 = weight_variable([10, n_output])
  b_fc3 = bias_variable([n_output])
  y_conv= tf.nn.softmax(tf.matmul(h_fc3, W_fc3) + b_fc3)
  # error = tf.reduce_mean(tf.abs(y_ - y_conv))
  error = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

with tf.device('/gpu:0'):
 train_step = tf.train.GradientDescentOptimizer(0.005).minimize(error)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
restore(saver, sess, '2dim_vgg_3class')

batch_size = 20 

for i in range(200000):
  batch_indices = np.random.choice(len(x_train), batch_size, replace=False) 
  batch_x, batch_y = fetch(x_train[batch_indices])
  fetches = [train_step, error, y_conv]
  t = sess.run(fetches, feed_dict={x: batch_x, y_: batch_y, drop: 0.5})
  
  miss = np.count_nonzero(np.argmax(t[2], axis=1) - np.argmax(batch_y, axis=1))
  print(i, t[1], miss)

  if i % 100 == 0:
    batch_x, batch_y = fetch(x_valid)
    fetches = [error, y_conv]
    t = sess.run(fetches, feed_dict={x: batch_x, y_: batch_y, drop: 0.5})
    miss = np.count_nonzero(np.argmax(t[1], axis=1) - np.argmax(batch_y, axis=1))
    print('VALIDATION: ', t[0], miss)

  
    # msg = '{} {}'.format(i, err)
    # os.system("curl \"https://api.telegram.org/bot236245101:AAFZ12aHX2emHeZuU99R11TdWMk9fZfl1j0/sendMessage?chat_id=237652977&text=" + msg + "\"")
    #print('')
  if i % 1000 == 0:
    saver.save(sess, "./tmp/model_2dim_vgg_3class.ckpt")
