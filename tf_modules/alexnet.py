from __future__ import print_function
from base_model import BaseModel
import tensorflow as tf
import numpy as np
import math


"""AlexNet model implementation in tensorflow.
    Paper:
        Alex Krizhevsky and Ilya Sutskever and Geoffrey E. Hinton.
        "Imagenet classification with deep convolutional neural networks."
        Advances in neural information processing systems. 2012.

    MODEL LAYOUT:
        ImageNet images resolution ->  downsampled to 256x256
        8 layers -> 5 convolutional and 3 fully connected
        Output layer -> 1000-way softmax
        multinomial logistic regression objective (maximaze the average)
        response-normalization layers after 1st and 2nd conv layers
        max-pooling layers after response-normalization layers and 5th conv layer
        Re-LU after every conv and fully connected layer

        max_pooling -> stride = 2, patch size = 3
        1st conv layer -> filter 224x224x3x48,  96 kernels 11x11x3, 4 pixels stride
        2nd conv layer -> filter 55x55x48x128,  256 kernels 5x5x48
        3rd conv layer -> filter 27x27x128x192, 384 kernels 3x3x256   
        4th conv layer -> filter 13x13x192x192, 384 kernels 3x3x192
        5th conv layer -> filter 13x13x192x128, 256 kernels 3x3x192
        fully connected layers - > 4096 neurons each
        NUM_LABELS = 1000
        NUM_NEURONS = 4096

        + overlapping pooling: stride = 2, patch size = 3
        + data augmentation
        + drop-out
"""


class AlexNet( BaseModel ):
    def __init__( self, dataset ):
        self.validation_data = dataset.validation_data
        self.validation_labels = dataset.validation_labels 
        self.train_data = dataset.train_data
        self.train_labels = dataset.train_labels
        test_data = dataset.test_data
        test_labels = dataset.test_labels

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

        print('----------------------------' )
        print('-- AlexNet --' )
        print('----------------------------' )
        print('Validation data: ', dataset.validation_data.shape)
        print('Validation labels: ', dataset.validation_labels.shape)
        print('Train_data:', self.train_data.shape)
        print('Train_labels:', self.train_labels.shape)
        print('Test_data:', dataset.test_data.shape)
        print('Test_labels:', dataset.test_labels.shape)
        print('Batch size:', self.BATCH_SIZE)
        print('Number of epochs:', self.NUM_EPOCHS)
        print('Evaluation batch size:', self.EVAL_BATCH_SIZE)
        print('Evaluation frequency:', self.EVAL_FREQUENCY)

        # "Declare" variables
        self.train_data_node = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.NUM_ROWS, self.NUM_COLS , self.NUM_CHANNELS))
        self.train_labels_node = tf.placeholder(tf.int32, shape=(self.BATCH_SIZE,))

        self.eval_data = tf.placeholder(tf.float32, shape=(self.EVAL_BATCH_SIZE, self.NUM_ROWS, self.NUM_COLS, self.NUM_CHANNELS))

        self.conv1_weights = tf.Variable( tf.truncated_normal([11, 11, self.NUM_CHANNELS, 48], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.conv1_biases = tf.Variable( tf.zeros([48], dtype=tf.float32) )

        self.conv2_weights = tf.Variable( tf.truncated_normal([5, 5, 48, 256], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.conv2_biases = tf.Variable( tf.constant(0.1, shape=[256], dtype=tf.float32) )
        
        self.conv3_weights = tf.Variable( tf.truncated_normal([3, 3, 256, 192], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.conv3_biases = tf.Variable( tf.constant(0.1, shape=[192], dtype=tf.float32) )
        
        self.conv4_weights = tf.Variable( tf.truncated_normal([3, 3, 192, 192], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.conv4_biases = tf.Variable( tf.constant(0.1, shape=[192], dtype=tf.float32) )

        self.conv5_weights = tf.Variable( tf.truncated_normal([3, 3, 192, 128], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.conv5_biases = tf.Variable( tf.constant(0.1, shape=[128], dtype=tf.float32) )

        # Layer 1 - First conv + max_pool
        row1 = math.ceil(self.NUM_ROWS / 2.)
        col1 = math.ceil(self.NUM_COLS / 2.)
        # Layer 2 - Second conv + max_pool
        row2 = math.ceil(row1 / 2.)
        col2 = math.ceil(col1 / 2.)
        # Layer 5 - Third conv + max_pool
        row3 = math.ceil(row2 / 2.)
        col3 = math.ceil(col2 / 2.)
        # Layer 5 - output feature map size
        feature_map_size = int(row3 * col3) * 128

        self.fc1_weights = tf.Variable( tf.truncated_normal([ feature_map_size, 4096], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.fc1_biases = tf.Variable( tf.constant(0.1, shape=[4096], dtype=tf.float32) )

        self.fc2_weights = tf.Variable( tf.truncated_normal( [4096, 4096], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.fc2_biases = tf.Variable( tf.constant(0.1, shape=[4096], dtype=tf.float32) )

        self.fc3_weights = tf.Variable( tf.truncated_normal( [4096, self.NUM_LABELS], stddev=0.1, seed=self.SEED, dtype=tf.float32) )
        self.fc3_biases = tf.Variable( tf.constant(0.1, shape=[self.NUM_LABELS], dtype=tf.float32) )

    def connected_layers( self, data, train = False ):
        conv_strides = [1, 1, 1, 1]
        pooling_strides = [1, 2, 2, 1]
        pooling_kernel = [1, 3, 3, 1]

        conv = tf.nn.conv2d( data, self.conv1_weights, conv_strides, padding='SAME' )
        relu = tf.nn.relu( tf.nn.bias_add(conv, self.conv1_biases) )
        norm = tf.nn.local_response_normalization( relu )
        pool = tf.nn.max_pool( relu, pooling_kernel, pooling_strides, padding='SAME' )

        conv = tf.nn.conv2d( pool, self.conv2_weights, conv_strides, padding='SAME' )
        relu = tf.nn.relu( tf.nn.bias_add(conv, self.conv2_biases) )
        norm = tf.nn.local_response_normalization( relu )
        pool = tf.nn.max_pool( relu, pooling_kernel, pooling_strides, padding='SAME' )

        conv = tf.nn.conv2d( pool, self.conv3_weights, conv_strides, padding='SAME' )
        relu = tf.nn.relu( tf.nn.bias_add(conv, self.conv3_biases) )

        conv = tf.nn.conv2d( relu, self.conv4_weights, conv_strides, padding='SAME' )
        relu = tf.nn.relu( tf.nn.bias_add(conv, self.conv4_biases) )

        conv = tf.nn.conv2d( relu, self.conv5_weights, conv_strides, padding='SAME' )
        relu = tf.nn.relu( tf.nn.bias_add(conv, self.conv5_biases) )
        pool = tf.nn.max_pool( relu, pooling_kernel, pooling_strides, padding='SAME' )


        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape( pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        hidden1 = tf.nn.relu( tf.matmul(reshape, self.fc1_weights) + self.fc1_biases )
        if train:
            hidden1 = tf.nn.dropout( hidden1, 0.25, seed=self.SEED )

        hidden2 = tf.nn.relu( tf.matmul(hidden1, self.fc2_weights) + self.fc2_biases )
        if train:
            hidden2 = tf.nn.dropout(hidden2, 0.25, seed=self.SEED )

        return tf.matmul(hidden2, self.fc3_weights) + self.fc3_biases

    def define_logits( self ):
        # Logits + cross-entropy loss.
        self.logits = self.connected_layers( self.train_data_node, train = True )
        self.loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.train_labels_node) )
        self.regularizers = ( tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                         tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases) +
                         tf.nn.l2_loss(self.fc3_weights) + tf.nn.l2_loss(self.fc3_biases) )
        self.loss += 5e-4 * self.regularizers

    def define_optimizer( self ):
        self.batch = tf.Variable( 0, dtype=tf.float32 )
        self.learning_rate = tf.train.exponential_decay( 0.01, self.batch * self.BATCH_SIZE, self.TRAIN_SIZE, 0.95, staircase=True )
        self.optimizer = tf.train.MomentumOptimizer( self.learning_rate, 0.9) .minimize( self.loss, global_step=self.batch )
        self.train_prediction = tf.nn.softmax( self.connected_layers(self.train_data_node, train = True) )
        self.eval_prediction = tf.nn.softmax( self.connected_layers(self.eval_data, train = False) )
