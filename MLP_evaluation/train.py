import tensorflow as tf
import numpy as numpy
import sys
import getopt
import csv as csv
import os as os
import matplotlib.pyplot as plt

from batch_feed import RandomBatchFeeder

class Train:
  def __init__(self, num_epochs,  learning_rate, size_hidden_layer, train_x, train_y):

    self.num_epochs = num_epochs
    self.learning_rate = learning_rate
    self.size_hidden_layer = size_hidden_layer
    self.train_x = train_x
    self.train_y = train_y

    # Current session
    self.sess = ""
    random_seed = 101  # used for all random initializations, to get reproducible results

    self.train_data_feeder = RandomBatchFeeder(train_x, train_y, random_seed)

    # --- Other Parameters --- #
    self.batch_size = 100  # set to 1 for single-sample training

    # --- Fixed Parameters --- #
    size_input_layer = 784  # dimension of a flattened MNIST image
    size_output_layer = 10  # one-hot output vector with 10 classes

    # --- Model Architecture --- #
    tf.set_random_seed(random_seed)

    # placeholder for input value batches: each row will contain one (flattened)
    # MNIST image, num rows will later be defined by batch size (None for now)
    x = tf.placeholder(tf.float32, [None, size_input_layer])
    self.x = x

    W_1 = tf.Variable(tf.random_uniform([size_input_layer, self.size_hidden_layer]))  # weights input to hidden
    b_1 = tf.Variable(tf.random_uniform([self.size_hidden_layer]))  # biases hidden

    W_2 = tf.Variable(tf.random_uniform([self.size_hidden_layer, size_output_layer]))  # weights hidden to output
    b_2 = tf.Variable(tf.random_uniform([size_output_layer]))  # biases output

    # Matrix with dimension [batch_size x size_hidden_layer], each row represents the hidden layer for one input row.
    # Use softmax as activation function.
    hidden_layer = tf.nn.softmax(tf.matmul(x, W_1) + b_1)

    # Matrix with dimension [batch_size x size_output_layer], each row represents the output layer for one input row.
    # Use softmax as activation function.
    output_layer = tf.nn.softmax(tf.matmul(hidden_layer, W_2) + b_2)

    # --- Learning --- #

    # Matrix with dimension [batch_size x size_output_layer], each line contains a one-hot encoded class label
    # for one input row
    y_labels = tf.placeholder(tf.float32, [None, size_output_layer])
    self.y_labels = y_labels

    # Cross-entropy cost, total loss for one input-batch
    cost_function = -tf.reduce_sum(y_labels * tf.log(output_layer))

    # Gradient descent optimizer, will minimize cost_function at learning_rate by optimizing all variables
    # in the connected Tensorflow graph of cost_function (or all vars in the default graph? Not sure...)
    self.optimizer = tf.train.GradientDescentOptimizer( self.learning_rate ).minimize(cost_function)

    # Tensorflow graph evaluation session - interactive session: installs itself as default session for run operations,
    # to avoid having to pass an explicit session object (not needed, but convenient for this prototype)
    self.sess = tf.InteractiveSession()

    # Initialize all variables in this Tensorflow graph (boilerplate, always done before graph evaluation)
    # -> tf.global_variables_initializer() is a TF graph node and must be run by the session (here, sess as default)
    tf.global_variables_initializer().run()

    # --- Model Evaluation ---
    # define model evaluation before learing so it can be used for cross-validation

    # take argmax of each row of output layer matrix and each one-hot encoded label row (i.e. find index of highest neuron
    # in output layer for each input row -> is the prediction class for that input row), compare each element of these
    # vectors for equality
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_labels, 1))
    self.prediction = tf.argmax(output_layer,1)

    # cast resulting boolean vector to float32 vector (tf.cast()), and calculate mean of its entries (tf.reduce_mean()).
    # Matches in the casted equality vector are 1, mismatches are 0, mean is percentage of correct predictions.
    # Cast from bool to numeric value is necessary to compute mean, use float32 to avoid rounding errors
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self.accuracy = accuracy

  def learn( self ):
    # Learning process: for num_epochs, run data in num_batches many batches and minimize cost_function
    for epoch in range( self.num_epochs ):
        total_loss = 0
        num_batches = int(self.train_x.shape[0] / self.batch_size)

        # for every batch, run optimizer, total loss for that batch
        for i in range(num_batches):
            batch_xs, batch_ys = self.train_data_feeder.next_batch(self.batch_size)  # get next input- and labels batch

            # Optimize model with the current batch
            self.sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y_labels: batch_ys})

  def classify(self, data):
    return self.prediction.eval(feed_dict={self.x: data }, session= self.sess)
