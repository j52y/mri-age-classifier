import os
import math
import itertools
import numpy as np
import tensorflow as tf
from utils import *
from scipy.ndimage.interpolation import zoom


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.025)
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


def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], 
                        strides=[1, 2, 2, 2, 1], padding='SAME')


def convl(l, r1, r2):
  W_conv = weight_variable([3, 3, 3, r1, r2])
  b_conv = bias_variable([r2])
  h_conv = tf.nn.relu(conv3d(l, W_conv) + b_conv)
  return h_conv


def vgg3d(x, d0, d1, d2):
  d = d0 * d1 * d2
  x_image = tf.reshape(x, [-1, d0, d1, d2, 1])

  n_pool = 6
  dt = 2 ** n_pool
  drow = math.ceil(d0/dt) * math.ceil(d1/dt) *  math.ceil(d2/dt) * 2048 

  with tf.device('/gpu:0'):
    conv1a = convl(x_image, 1, 64) 
    conv1b = convl(conv1a, 64, 64)
    h_pool1 = max_pool_2x2x2(conv1b)
    
    conv2a = convl(h_pool1, 64, 128)
    conv2b = convl(conv2a, 128, 128)
    h_pool2 = max_pool_2x2x2(conv2b)
  
    conv3a = convl(h_pool2, 128, 256)
    conv3b = convl(conv3a, 256, 256)
    h_pool3 = max_pool_2x2x2(conv3b) 
  
    conv4a = convl(h_pool3, 256, 512)
    conv4b = convl(conv4a, 512, 512)
#    conv4c = convl(conv4b, 512, 512)
    h_pool4 = max_pool_2x2x2(conv4b)
  
    conv5a = convl(h_pool4, 512, 1024)
    conv5b = convl(conv5a, 1024, 1024)
#    conv5c = convl(conv5b, 1024, 1024)
    h_pool5 = max_pool_2x2x2(conv5b)

    conv6a = convl(h_pool5, 1024, 2048)
    conv6b = convl(conv6a, 2048, 2048)
#    conv6c = convl(conv6b, 2048, 2048)
    h_pool6 = max_pool_2x2x2(conv6b)

    h_pool6_flat = tf.reshape(h_pool6, [-1, drow])
#  conv7a = convl(h_pool6, 2048, 4096)
#  conv7b = convl(conv7a, 4096, 4096)
#  conv7c = convl(conv7b, 4096, 4096)
#  h_pool7 = max_pool_2x2x2(conv7c)

  return h_pool6_flat, drow

ckpt_name = '3d_vgg_3cls'
def restore(saver, sess, name=ckpt_name):
  fname = "./tmp/model_" + name + ".ckpt"
  if os.path.isfile(fname):
    saver.restore(sess, fname)

younger = 25
older = 72.14
padding = 20.38

d0, d1, d2 = 192, 192, 100 #512, 512, 100
d = d0 * d1 * d2

n_output = 3

def fetch(keys):
  fetch_x = [normalize_image(crop_image(imgs[k].get_data(), [d0, d1, d2])).reshape(-1) for k in keys]
  fetch_y = [[1,0,0] if ages[k] <= younger else [0,0,1] if ages[k] > older else [0,1,0] for k in keys]
  return fetch_x, fetch_y


def divide_set(keys):
  trains = [k for i, k in enumerate(keys) if i % 9 != 0]
  valids = [k for i, k in enumerate(keys) if i % 9 == 0]
  return trains, valids


imgs, ages = load_ixi_data()

youngs = [k for k, v in ages.items() if v <= younger]
middles = [k for k, v in ages.items() if v >= younger + padding and v < older - padding]
olds = [k for k, v in ages.items() if v > older]

print("The Number of Samples per Class: ", len(youngs), len(middles), len(olds))


young_train, young_valid = divide_set(youngs)
middle_train, middle_valid = divide_set(middles)
old_train, old_valid = divide_set(olds)


x_train = np.vstack((young_train, middle_train, old_train)).reshape((-1,),order='F') #interweave
x_valid = np.array(young_valid + middle_valid + old_valid)

#  def fit_group_size(group, size):
#    l = len(group)
#    if l >= size:
#      return group[:size]
#    return fit_group_size(group + group, size)
#  
#  
#  def age_group(ages):
#    g20 = [k for k,v in ages.items() if v < 30]
#    g30 = [k for k,v in ages.items() if 30 <= v < 40]
#    g40 = [k for k,v in ages.items() if 40 <= v < 50]
#    g50 = [k for k,v in ages.items() if 50 <= v < 60]
#    g60 = [k for k,v in ages.items() if 60 <= v < 70]
#    g70 = [k for k,v in ages.items() if 70 <= v < 80]
#    g80 = [k for k,v in ages.items() if 80 <= v < 90]
#    return list(itertools.chain.from_iterable(zip(fit_group_size(g20, 100),fit_group_size(g30, 100),fit_group_size(g40, 100),fit_group_size(g50, 100),fit_group_size(g60, 100),fit_group_size(g70, 100),fit_group_size(g80, 100))))
# 

x = tf.placeholder(tf.float32, shape=[None, d])
y_ = tf.placeholder(tf.float32, shape=[None, n_output])
drop = tf.placeholder(tf.float32)
h_vgg, r2 = vgg3d(x, d0, d1, d2)

# with tf.device('/cpu:0'):

with tf.device('/cpu:0'):
  h_fc0 = fc2d(h_vgg, r2, 4096, drop)

with tf.device('/gpu:0'):
  h_fc1 = fc2d(h_fc0, 4096, 1024, drop)
  h_fc2 = fc2d(h_fc1, 1024, 128, drop)
  h_fc3 = fc2d(h_fc2, 128, 10, drop)
  
  W_fc4 = weight_variable([10, n_output])
  b_fc4 = bias_variable([n_output])
  y_conv= tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)

  error = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
  train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(error)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
restore(saver, sess)

error_sum = 0.0
for i in range(2000000):
  j = i%len(x_train)
  batch_x, batch_y = fetch([x_train[j]]) 
  fetches = [train_step, error, y_conv]
  t = sess.run(fetches, feed_dict={x: batch_x, y_: batch_y, drop: 0.5})
  err = t[1]
  error_sum += err
  pred = t[2]
  print(i, j, batch_y[0], pred[0], err, error_sum)

  if j == len(x_train)-1:
    saver.save(sess, "./tmp/model_" + ckpt_name + ".ckpt")
    error_sum = 0.0

    for k in range(len(x_valid)):
      batch_x, batch_y = fetch([x_valid[k]])
      fetches = [error, y_conv]
      t = sess.run(fetches, feed_dict={x: batch_x, y_: batch_y, drop: 0.5})
      err = t[0]
      pred = t[1]
      print('VALIDATION', k, batch_y[0], pred[0], err)

      msg = '{} {} {} {}'.format(k, err, np.array(batch_y[0]).argmax(), np.array(pred[0]).argmax())
      os.system("curl \"https://api.telegram.org/bot236245101:AAFZ12aHX2emHeZuU99R11TdWMk9fZfl1j0/sendMessage?chat_id=237652977&text=" + msg + "\"")
      print('')
