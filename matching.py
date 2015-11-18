# these imports I have no idea what they do but whatever
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow.python.platform

# import these, they're useful
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

# my data generation
import data_gen as dg

# the only length this model can handle before gradiant vanishes
example_length = 8

# initialize stuff

# the size for the memory unit within a lstm
num_units = 20
# the output dim for the lstm (this isn't used explicitly in this model)
num_proj = 20
# the state is both the memory and the output (for this particular lstm)
state_size = num_units + num_proj
# running over 100 examples at a time
batch_size = 100
# each character is a 1-hot encoding for "(", ")", "0"
input_size = 3
# the output is a 1-hot encoding for "true" or "false"
label_size = 2

# post proccessing layer
hidden_units = 50

sess = tf.Session()
with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)) as scope:

  # ================ MODEL CONSTRUCTION ================== #
  # our cell
  cell = rnn_cell.LSTMCell(num_units=num_units, 
                           input_size=input_size, 
                           num_proj=num_proj)
  
  # initial state
  state_init = tf.Variable(tf.zeros([1, state_size]))
  state = tf.tile(state_init, [batch_size, 1])

  # input seq, i.e. [(, ), (, (, ), ), ...0, 0], 1-hot encoded
  input_seq = tf.placeholder(tf.float32, [None, example_length, input_size])
  # output label, i.e. True / False, 1-hot encoded
  output_label_ = tf.placeholder(tf.float32, [None, label_size])

  # fixed number of unrollings and channing of lstm units
  for i in range(0, example_length):
    if i > 0:
      scope.reuse_variables()
    out, state = cell(input_seq[:, i, :], state)

  # post proccessing of the final state of the lstm, first run it through a relu unit
  w1 = tf.Variable(tf.random_normal([state_size, hidden_units], mean=0.1, stddev=0.035))
  b1 = tf.Variable(tf.zeros([hidden_units]))
  relu1 = tf.nn.relu(tf.matmul(state, w1) + b1)

  # then output a soft-max of the 1-hot encoding for true/false
  w2 = tf.Variable(tf.random_normal([hidden_units, label_size], mean=0.1, stddev=0.035))
  b2 = tf.Variable(tf.zeros([label_size]))
  output_label = tf.nn.softmax(tf.matmul(relu1, w2) + b2)

  # minimize cross entropy against the true label
  cross_entropy = -tf.reduce_sum(output_label_*tf.log(output_label))

  # get gradients and clip them, use adaptive to prevent explosions
  tvars = tf.trainable_variables()
  grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(cross_entropy, tvars)]
  optimizer = tf.train.AdagradOptimizer(0.01)
  train_step = optimizer.apply_gradients(zip(grads, tvars))


  # =============== TRAINING AND TESTING ================== #
  # initialize
  sess.run([tf.variables.initialize_all_variables()])
  # run over many epochs
  for i in range(50001):
    # get data dynamically from my data generator
    dat, lab = dg.gen_data_batch(batch_size, example_length)
    # do evaluation every 100 epochs
    if (i % 100 == 0):
      print("====current accuracy==== at epoch ", i)
      pos_data, pos_label = dg.gen_data_batch(100, example_length, pos_neg = True)
      neg_data, neg_label  = dg.gen_data_batch(100, example_length, pos_neg = False)
      correct_prediction = tf.equal(tf.argmax(output_label,1), tf.argmax(output_label_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      res = sess.run([accuracy], feed_dict={input_seq: pos_data, output_label_: pos_label})
      print("pos accuracy: ", res)
      res = sess.run([accuracy], feed_dict={input_seq: neg_data, output_label_: neg_label})
      print("neg accuracy: ", res)
    # continuously train at every epoch
    sess.run(train_step, feed_dict={input_seq: dat, output_label_: lab})

