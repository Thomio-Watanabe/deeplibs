# CNN Model created from udacity deep learning course
import tensorflow as tf


def model(images_info, data_size, input_data):
    nrows = images_info[0]
    ncols = images_info[1]
    num_channels = images_info[2]
    output_size = images_info[3]
    npixels = nrows * ncols
    nimages_training = data_size[0]
    images_training = input_data[0]
    bb_training = input_data[1]


    batch_size = 10
    iterations = int( nimages_training / batch_size )  # steps, epochs
    learning_rate = 0.05
    
    # convolution stuff
    patch_size_row = 25
    patch_size_col = 27
    depth = 16 # 128
    image_size = npixels
    num_hidden = 32 # 256
    
    
    tf_training_images = tf.placeholder( tf.float32, shape = (batch_size, nrows, ncols, num_channels) )
    tf_training_bb = tf.placeholder( tf.float32, shape = (batch_size, output_size) )
    # tf_validation_images
    # tf_test_images

    # NN weigths
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size_row, patch_size_col, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size_row, patch_size_col, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    
    layer3_weights = tf.Variable(tf.truncated_normal( [ 43200, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[output_size]))

    # weigths = tf.Variable( tf.truncated_normal( [npixels, output_size] ) )
    # biases = tf.Variable( tf.zeros([output_size]) )

    def connections(data):
        strides = [1, 2, 2, 1]
        conv = tf.nn.conv2d(data, layer1_weights, strides, padding='SAME')
        # tf.nn.batch_normalization()
        relu = tf.nn.relu(conv + layer1_biases)
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, layer2_weights, strides, padding='SAME')
        relu = tf.nn.relu(conv + layer2_biases)
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = connections(tf_training_images)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_training_bb))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_training_bb))
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer( learning_rate ).minimize(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print( 'Training CNN...' )
        for i in range( iterations ):
            offset = (i * batch_size) % (nimages_training - batch_size)
            batch_data = images_training[ offset:(offset + batch_size), :]
            batch_bb = bb_training[ offset:(offset + batch_size), :]
            _, l = sess.run( [optimizer, loss], feed_dict={tf_training_images: batch_data, tf_training_bb: batch_bb} )
            print( 'Minibatch loss at step', i,'=', l )
    #        if( i % batch_size == 0):
    #            print( 'Minibatch loss at step', i,'=', l )

