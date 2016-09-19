import numpy as np
import tensorflow as tf
from utils import *

r_range = 0.1
i_max = 1480
train_x, train_y = load_train_data()

min_age, max_age = min(train_y), max(train_y)
input_size = 360 * 512 
output_size = 1 # len(set(train_y))

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, output_size])

W0 = tf.Variable(tf.random_uniform((input_size, 100), -r_range, r_range), name='W')
b0 = tf.Variable(tf.random_uniform((100,), -r_range, r_range), name='b')

W1 = tf.Variable(tf.random_uniform((100, output_size), -r_range, r_range), name='W')
b1 = tf.Variable(tf.random_uniform((output_size,), -r_range, r_range), name='b')

sess.run(tf.initialize_all_variables())

h0 = tf.matmul(X, W0) + b0
y0 = tf.nn.sigmoid(h0)

h1 = tf.matmul(y0, W1) + b1
y1 = tf.nn.sigmoid(h1)

cost = tf.reduce_sum(tf.abs(tf.sub(y1, y_)))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

for i in range(1000):
  for j in range(len(train_x)):
    print(i, j)
    if train_x[j].shape[0] * train_x[j].shape[1] != input_size:
      continue

    batch_x = np.max(train_x[j].get_data(), axis=2).reshape(1, input_size) / i_max
    batch_y = np.array([[train_y[j]/(max_age - min_age)]]) 
    optimizer.run(feed_dict={X: batch_x, y_: batch_y})

  s = 0
  for j in range(len(train_x)):
    if train_x[j].shape[0] * train_x[j].shape[1] != input_size:
      continue

    batch_x = np.max(train_x[j].get_data(), axis=2).reshape(1, input_size) / i_max
    batch_y = np.array([[(train_y[j] - min_age) / (max_age - min_age)]]) 
    s += sess.run(cost, feed_dict={X: batch_x, y_: batch_y})
  
  print(s)

   

