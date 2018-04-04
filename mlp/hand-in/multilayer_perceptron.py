import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as numpy
from data_loader import load_data
from batch_feed import RandomBatchFeeder


# --- Hyperparameters --- #
num_epochs = 40
learning_rate = 0.1
size_hidden_layer = 60


# --- Other Parameters --- #
batch_size = 100  # set to 1 for single-sample training
random_seed = 101  # used for all random initializations, to get reproducible results


# --- Fixed Parameters --- #
size_input_layer = 784  # dimension of a flattened MNIST image
size_output_layer = 10  # one-hot output vector with 10 classes


train_x, train_y = load_data("data/train.csv") # first 20'000 lines of origin train set
train_data_feeder = RandomBatchFeeder(train_x, train_y, random_seed)
test_x, test_y = load_data("data/test.csv")
valid_x, valid_y = load_data("data/valid.csv") # last 6'999 lines of origin train set

# --- Model Architecture --- #
tf.set_random_seed(random_seed)

# placeholder for input value batches: each row will contain one (flattened)
# MNIST image, num rows will later be defined by batch size (None for now)
x = tf.placeholder(tf.float32, [None, size_input_layer])

W_1 = tf.Variable(tf.random_uniform([size_input_layer, size_hidden_layer]))  # weights input to hidden
b_1 = tf.Variable(tf.random_uniform([size_hidden_layer]))  # biases hidden

W_2 = tf.Variable(tf.random_uniform([size_hidden_layer, size_output_layer]))  # weights hidden to output
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

# Cross-entropy cost, total loss for one input-batch
cost_function = -tf.reduce_sum(y_labels * tf.log(output_layer))

# Gradient descent optimizer, will minimize cost_function at learning_rate by optimizing all variables
# in the connected Tensorflow graph of cost_function (or all vars in the default graph? Not sure...)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Tensorflow graph evaluation session - interactive session: installs itself as default session for run operations,
# to avoid having to pass an explicit session object (not needed, but convenient for this prototype)
sess = tf.InteractiveSession()

# Initialize all variables in this Tensorflow graph (boilerplate, always done before graph evaluation)
# -> tf.global_variables_initializer() is a TF graph node and must be run by the session (here, sess as default)
tf.global_variables_initializer().run()

# --- Error Rate plot
# necessary arrays to make the plot of the error-rate on training and validation set wirth respect to the number of training epochs
training_epochs = [] # stores the number of epoche
accurancy_train_set = []
accurancy_valid_set = []

# --- Model Evaluation ---
# define model evaluation before learing so it can be used for cross-validation

# take argmax of each row of output layer matrix and each one-hot encoded label row (i.e. find index of highest neuron
# in output layer for each input row -> is the prediction class for that input row), compare each element of these
# vectors for equality
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_labels, 1))

# cast resulting boolean vector to float32 vector (tf.cast()), and calculate mean of its entries (tf.reduce_mean()).
# Matches in the casted equality vector are 1, mismatches are 0, mean is percentage of correct predictions.
# Cast from bool to numeric value is necessary to compute mean, use float32 to avoid rounding errors
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Learning process: for num_epochs, run data in num_batches many batches and minimize cost_function
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = int(train_x.shape[0] / batch_size)
    training_epochs.append(epoch)
    # for every batch, run optimizer, total loss for that batch
    for i in range(num_batches):
        batch_xs, batch_ys = train_data_feeder.next_batch(batch_size)  # get next input- and labels batch
        sess.run(optimizer, feed_dict={x: batch_xs, y_labels: batch_ys})  # run optimizer for new batch

        # calculate loss of that batch, just as a reference
        total_loss += sess.run(cost_function, feed_dict={x: batch_xs, y_labels: batch_ys})

    # print the accuracy on training and validation set after each training epoch
    # and store it in an array
    accurancy_train_set.append(sess.run(accuracy, feed_dict={x: train_x, y_labels: train_y}))
    accurancy_valid_set.append(sess.run(accuracy, feed_dict={x: valid_x, y_labels: valid_y}))

    # print average loss over all batches in that epoch, as a reference
    print("Epoch {} complete, loss={}".format(epoch + 1, total_loss/num_batches))

# for cross-validation: stores the plot of error-rate on training and validation set with respect to training epochs
training_epochs = numpy.array(training_epochs) # change List to Array
accurancy_train_set = numpy.array(accurancy_train_set)
accurancy_valid_set = numpy.array(accurancy_valid_set)
plt.plot(training_epochs, accurancy_train_set, label='accuracy for train set'.format(i=1))
plt.plot(training_epochs, accurancy_valid_set, label='accuracy for validation set'.format(i=2))
plt.legend(loc='best')
plt.xlabel('training epoch number')
plt.suptitle('Plot with learning rate ' + str(learning_rate) + " and " + str(size_hidden_layer) + " hidden layers.")
print("Plot with error-rate of training and validation set depending on training epoch is made (for cross-validation).")
plt.savefig('Plot with learning rate ' + str(learning_rate) + " and " + str(size_hidden_layer) + " hidden layers.")

# for cross-validation: store learning rate, number of hidden layers and accurancy of validation set in txt file
file = open("cross-validation values.txt", "a")
file.write("For cross-validation. This is the accuracy for validation set with " + str(num_epochs) + " number of epochs, learning rate " + str(learning_rate) + " and " + str(size_hidden_layer) + " hidden layers:\n")
file.write(str(sess.run(accuracy, feed_dict={x: valid_x, y_labels: valid_y}))+ "\n\n")
file.close()

# run this eval on the MNIST test set and test labels and print result
print("This is the final accuracy for test set after the whole procedure. This value has only to be read in the end for the optimized parameters.")
print(sess.run(accuracy, feed_dict={x: test_x, y_labels: test_y}))
