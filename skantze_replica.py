from __future__ import print_function

import tensorflow as tf
import data_reader
import numpy as np
from tensorflow.contrib import rnn

# Hyper-params
learning_rate = 0.001
training_steps = 1200
batch_size = 128
display_step = 1
beta = 0.001

num_input = 6 * 2  # Prosody
timesteps = 1200  # 60 sec * 20 frames/sec = 1200
num_hidden = 10  # num units in LSTM cell
experiments = [5,10,20,40,60]

for experiment in experiments:

    tf.reset_default_graph()
    num_output_units = experiment  # 20 frames/sec

    # Reading data
    print("Reading data...")
    x_train, y_train, x_test, y_test = data_reader.get_data()
    y_train = y_train[:,:,0:num_output_units]
    y_test = y_test[:,:,0:num_output_units]
    print(x_train.shape)

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, timesteps, num_output_units])

    # Define weights/biases
    weights = {
        'out': tf.get_variable("w_out", shape=[num_hidden, num_output_units],
               initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'out': tf.get_variable("b_out", shape=[num_output_units],
               initializer=tf.contrib.layers.xavier_initializer())
    }


    def build_lstm_rnn(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, num_input)
        # Required structure: list of size 'timesteps', where each item in the list is a tensor of shape:
        #                                                                           (batch_size, num_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, timesteps, 1)

        # Basic LSTM Cell with num_hidden units
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Skantze used a static structure instead of a dynamic one. Why?
        # outputs is a list of tensors of shape (batch_size, num_hidden). Size of the list: timesteps
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Reshape 'outputs' to be a 2D matrix, so we can perform outputs * weights
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # sigmoid_acts = tf.sigmoid(tf.add(tf.matmul(outputs, weights['out']), biases['out']))
        sigmoid_acts = tf.sigmoid(tf.add(tf.matmul(outputs, weights['out']), biases['out']))

        # Reshape the result back to (timesteps, num_examples, num_output_units)
        sigmoid_acts = tf.reshape(sigmoid_acts, [timesteps, -1, num_output_units])

        # The shape of the prediction must be (num_examples, timesteps, num_output_units)
        sigmoid_acts = tf.transpose(sigmoid_acts, [1,0,2])

        return sigmoid_acts


    print("Building graph...")
    net_out = build_lstm_rnn(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.squared_difference(net_out, Y))
    regularizer = tf.nn.l2_loss(weights['out'])
    loss_op = tf.reduce_mean(loss_op + beta * regularizer)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Mean Absolute Error operation to measure performance
    mae_op = tf.reduce_mean(tf.abs(Y - tf.minimum(tf.maximum(net_out, 0), 1)))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    print("Starting session... Num output units: ", experiment)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    # Start training
    with tf.Session() as sess:
        # saver.restore(sess, "toyota_sigmoid_model" + str(experiment) + ".ckpt")

        # Run the initializer
        sess.run(init)
        num_batches = int(x_train.shape[0] / batch_size)

        for epoch in range(training_steps):
            for step in range(num_batches):
                idx = np.random.randint(x_train.shape[0], size=batch_size)
                batch_x = x_train[idx, :, :]
                batch_y = y_train[idx, :, :]

                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            epoch_mae = sess.run([mae_op], feed_dict={X: x_test, Y: y_test})

            # epoch_cost /= num_batches
            print("Epoch " + str(epoch) + ", Epoch MAE= " + str(epoch_mae))

        print("Optimization Finished!")

        save_path = saver.save(sess, "./Skantze_Replica_Model_" + str(experiment) + ".ckpt")
        print("Model saved in file: %s" % save_path)
