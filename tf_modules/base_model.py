import tensorflow as tf
import numpy as np
import time
import sys


'''
Source code retrived from tensorflow: https://github.com/tensorflow/tensorflow
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


class BaseModel:
    def error_rate(self, predictions, labels):
        """Return the error rate based on dense predictions and sparse labels."""
        return 100.0 - ( 100.0 * np.sum(np.argmax(predictions, 1) == labels) /
                       predictions.shape[0])

    def eval_in_batches(self, data, sess):
        size = data.shape[0]
        if size < self.EVAL_BATCH_SIZE:
            raise ValueError("Batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, self.NUM_LABELS), dtype=np.float32)
        for begin in range(0, size, self.EVAL_BATCH_SIZE):
            end = begin + self.EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run( self.eval_prediction, feed_dict={self.eval_data: data[begin:end, ...]} )
            else:
                batch_predictions = sess.run( self.eval_prediction, feed_dict={self.eval_data: data[-self.EVAL_BATCH_SIZE:, ...]} )
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def train( self ):
        self.define_logits()
        self.define_optimizer()
        start_time = time.time()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for step in range( int(self.NUM_EPOCHS * self.TRAIN_SIZE) // self.BATCH_SIZE ):
                offset = (step * self.BATCH_SIZE) % (self.TRAIN_SIZE - self.BATCH_SIZE)
                batch_data = self.train_data[ offset:(offset + self.BATCH_SIZE), ... ]
                batch_labels = self.train_labels[ offset:(offset + self.BATCH_SIZE) ]
                _, l, lr, predictions = sess.run( [self.optimizer, self.loss, self.learning_rate, self.train_prediction],
                                                  feed_dict={self.train_data_node: batch_data,
                                                             self.train_labels_node: batch_labels})
                if step % self.EVAL_FREQUENCY == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print( 'Step %d (epoch %.2f), %.1f ms' %
                           (step, float(step) * self.BATCH_SIZE / self.TRAIN_SIZE,
                            1000 * elapsed_time / self.EVAL_FREQUENCY) )
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % self.error_rate( predictions, batch_labels) )
                    print('Validation error: %.1f%%' % self.error_rate( self.eval_in_batches(self.validation_data, sess), self.validation_labels) )
                    sys.stdout.flush()
            test_error = self.error_rate( self.eval_in_batches(self.test_data, sess), self.test_labels )
            print('Test error: %.1f%%' % test_error)
            print('-- Finished training model.')

