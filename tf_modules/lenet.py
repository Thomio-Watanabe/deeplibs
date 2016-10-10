from __future__ import print_function
from base_model import BaseModel
import tensorflow as tf
import numpy as np
import math

'''
LeNet model retrived from tensorflow source code at: www.github.com/tensorflow
This code is under apache licence which is compatible with GNU GPLv2
Modified by Thomio Watanabe
'''
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""




class LeNet( BaseModel ):
  def __init__( self, dataset ):
    self.validation_data = dataset.validation_data
    self.validation_labels = dataset.validation_labels
    self.train_data = dataset.train_data
    self.train_labels = dataset.train_labels
    self.test_data = dataset.test_data
    self.test_labels = dataset.test_labels

    self.SEED = None  # Set to None for random seed
    self.NUM_ROWS = dataset.num_rows
    self.NUM_COLS = dataset.num_cols
    self.NUM_LABELS = dataset.num_labels
    self.NUM_CHANNELS = dataset.num_channels
    self.TRAIN_SIZE = dataset.train_labels.shape[0]
    self.BATCH_SIZE = dataset.batch_size
    self.NUM_EPOCHS = dataset.num_epochs
    self.EVAL_BATCH_SIZE = dataset.eval_batch_size
    self.EVAL_FREQUENCY = dataset.eval_frequency

    print('-----------' )
    print('-- LeNet --' )
    print('-----------' )
    print('Validation data: ', self.validation_data.shape)
    print('Validation labels: ', self.validation_labels.shape)
    print('Train_data:', self.train_data.shape)
    print('Train_labels:', self.train_labels.shape)
    print('Test_data:', self.test_data.shape)
    print('Test_labels:', self.test_labels.shape)
    print('Batch size:', self.BATCH_SIZE)
    print('Number of epochs:', self.NUM_EPOCHS)
    print('Evaluation batch size:', self.EVAL_BATCH_SIZE)
    print('Evaluation frequency:', self.EVAL_FREQUENCY)


    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    self.train_data_node = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.NUM_ROWS, self.NUM_COLS , self.NUM_CHANNELS))
    self.train_labels_node = tf.placeholder(tf.int32, shape=(self.BATCH_SIZE,))
    self.eval_data = tf.placeholder(tf.float32, shape=(self.EVAL_BATCH_SIZE, self.NUM_ROWS, self.NUM_COLS, self.NUM_CHANNELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    self.conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=self.SEED, dtype=tf.float32))
    self.conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))

    self.conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.1,
      seed=self.SEED, dtype=tf.float32))
    self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))

    # First conv + max_pool
    row1 = math.ceil(self.NUM_ROWS / 2.)
    col1 = math.ceil(self.NUM_COLS / 2.)
    # Second conv + max_pool
    row2 = math.ceil(row1 / 2.)
    col2 = math.ceil(col1 / 2.)
    # Layer 2 output feature map size
    feature_map_size = int(row2 * col2) * 64

    self.fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([ feature_map_size, 512],
                          stddev=0.1,
                          seed=self.SEED,
                          dtype=tf.float32))
    self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))

    self.fc2_weights = tf.Variable(tf.truncated_normal([512, self.NUM_LABELS],
                                                stddev=0.1,
                                                seed=self.SEED,
                                                dtype=tf.float32))
    self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS], dtype=tf.float32))



  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def connections(self, data, train = False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        self.conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        self.conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=self.SEED)
    return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

  def define_logits( self ):
    # Training computation: logits + cross-entropy loss.
    self.logits = self.connections(self.train_data_node, train = True)
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      self.logits, self.train_labels_node))

    # L2 regularization for the fully connected parameters.
    self.regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                  tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
    # Add the regularization term to the loss.
    self.loss += 5e-4 * self.regularizers

  def define_optimizer( self ):
    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    self.batch = tf.Variable(0, dtype=tf.float32)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    self.learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      self.batch * self.BATCH_SIZE,  # Current index into the dataset.
      self.TRAIN_SIZE,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
    # Use simple momentum for the optimization.
    self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=self.batch)

    # Predictions for the current training minibatch.
    self.train_prediction = tf.nn.softmax(self.logits)

    # Predictions for the test and validation, which we'll compute less often.
    self.eval_prediction = tf.nn.softmax( self.connections(self.eval_data) )

