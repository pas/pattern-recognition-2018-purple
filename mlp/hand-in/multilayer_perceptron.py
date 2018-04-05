import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as numpy
import sys
import getopt
import csv as csv
import os as os

from data_loader import load_data
from batch_feed import RandomBatchFeeder

try:
  opts, args = getopt.getopt(sys.argv[1:], "he:r:l:f:", ["help","test_file","num_epochs=","learning_rate=","num_folds="])
except getopt.GetoptError:
  print getopt.GetoptError.msg
  print 'multilayer_perceptron.py -e <number of Epochs> -r <learning Rate> -l <number of hidden Layers> -f <Number of folds>'
  sys.exit(2)

for opt, arg in opts:
  if opt in ("-h", "--help"):
    print 'multilayer_perceptron.py -e <number of Epochs> -r <learning Rate> -l <number of hidden Layers> -f <Fold number>'
    sys.exit()
  elif opt in ("-e", "--num_epochs"):
    num_epochs = int(arg)
  elif opt in ("-r", "--learning_rate"):
    learning_rate = float(arg)
  elif opt in ("-l", "--num_hidden_layers"):
    size_hidden_layer = int(arg)
  elif opt in ("-l", "--num_hidden_layers"):
    size_hidden_layer = int(arg)
  elif opt in ("-f", "--num_folds"):
    num_folds = int(arg)

##
# Plotting full results
##
class PlotFull:
  def __init__( self , num_epochs , learning_rate , size_hidden_layer , num_folds):    
    test = numpy.genfromtxt('results/results-test.csv', delimiter=',')
    mean_test = numpy.mean(test, axis=0)
    stdev_test = test.std(0)
    
    train = numpy.genfromtxt('results/results-train.csv', delimiter=',')  
    mean_train = numpy.mean(train, axis=0)     
    
    training_epochs = numpy.arange( 1, len(mean_train)+1 )
    
    plt.plot(training_epochs, mean_test, label='mean accuracy for test set'.format(i=1))
    plt.plot(training_epochs, mean_train, label='mean accuracy for train set'.format(i=2))
    plt.errorbar(training_epochs, mean_test, stdev_test, ecolor='grey', linestyle="None",  lw=1)
    
    plt.legend(loc='best')
    plt.xlabel('training epoch number')
    plt.suptitle('Plot of mean of all ' + str(num_folds) + ' folds with learning rate ' + str( learning_rate ) + " and " + str(size_hidden_layer) + " hidden layers.")
    plt.savefig('plots/full/Plot with learning rate ' + str( learning_rate ) + " and " + str(size_hidden_layer) + " hidden layers with " + str(num_folds) + " folds.")
    
    result_file = open("results.csv", "ab")
    writer = csv.writer(result_file, delimiter=",")
    writer.writerow( mean_train )
    writer.writerow( mean_test )
    result_file.close()
    

class Train:
  def __init__(self, num_epochs,  learning_rate, size_hidden_layer, leave_out_fold):
    print 'Number of epochs: ', num_epochs
    print 'Learning rate: ', learning_rate
    print 'Number of hidden layers: ', size_hidden_layer
    
    self.num_epochs = num_epochs
    self.learning_rate = learning_rate
    self.size_hidden_layer = size_hidden_layer
    self.leave_out_fold = leave_out_fold
    
    self.validation_file = "data/valid.csv"
    
    # final accuracy test with the validation set
    self.final_accuracy = 0
    
    # necessary arrays to make the plot of the error-rate on training and validation set wirth respect to the number of training epochs
    self.training_epochs = [] # stores the number of epoche
    self.accurancy_train_set = []
    self.accurancy_valid_set = []
    
    # Current session
    self.sess = ""

  def start( self ):
    # is this really a good idea? Should we not repeatedly do the initialization and
    # then take the mean?
    random_seed = 101  # used for all random initializations, to get reproducible results

    # This part expects that ther256e are files in the folds it loads the files except the left out
    # given as parameter

    train_x = numpy.array([])
    train_y = numpy.array([])
    empty = True

    for fold in range(0,4):
      fold_file = "folds/fold-"+str(fold)+".csv"
      
      if( fold != self.leave_out_fold ):
        current_x, current_y = load_data( fold_file )
        
        # Couldn't find a better solution to make this more slick
        if( empty ):
          train_x = current_x
          train_y = current_y
          empty = False
        else:
          train_x = numpy.concatenate( (train_x, current_x) , axis=0)
          train_y = numpy.concatenate( (train_y, current_y) , axis=0)
      else:
        sys.stdout.write( "Testing fold ---> " )
        test_x, test_y = load_data( fold_file )

    valid_x, valid_y = load_data( self.validation_file ) # last 6'999 lines of origin train set
    
    self.__learn( random_seed , train_x , train_y , test_x , test_y , valid_x , valid_y )

    # for cross-validation: stores the plot of error-rate on training and validation set with respect to training epochs
    self.training_epochs = numpy.array( self.training_epochs ) # change List to Array
    self.accurancy_train_set = numpy.array(self.accurancy_train_set)
    self.accurancy_valid_set = numpy.array(self.accurancy_valid_set)
    
    self.__write_to_file()
    self.__plot()

    # run this eval on the MNIST test set and test labels and print result
    print("This is the final accuracy for test set after the whole procedure. This value has only to be read in the end for the optimized parameters.")
    print( self.final_accuracy )

  ##
  #
  # Starts the learning phase
  #
  ##
  def __learn( self, random_seed , train_x , train_y , test_x, test_y, valid_x , valid_y ):
    train_data_feeder = RandomBatchFeeder(train_x, train_y, random_seed)
    
    # --- Other Parameters --- #
    batch_size = 100  # set to 1 for single-sample training
    
    # --- Fixed Parameters --- #
    size_input_layer = 784  # dimension of a flattened MNIST image
    size_output_layer = 10  # one-hot output vector with 10 classes
    
    # --- Model Architecture --- #
    tf.set_random_seed(random_seed)

    # placeholder for input value batches: each row will contain one (flattened)
    # MNIST image, num rows will later be defined by batch size (None for now)
    x = tf.placeholder(tf.float32, [None, size_input_layer])

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

    # Cross-entropy cost, total loss for one input-batch
    cost_function = -tf.reduce_sum(y_labels * tf.log(output_layer))

    # Gradient descent optimizer, will minimize cost_function at learning_rate by optimizing all variables
    # in the connected Tensorflow graph of cost_function (or all vars in the default graph? Not sure...)
    optimizer = tf.train.GradientDescentOptimizer( self.learning_rate ).minimize(cost_function)

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

    # cast resulting boolean vector to float32 vector (tf.cast()), and calculate mean of its entries (tf.reduce_mean()).
    # Matches in the casted equality vector are 1, mismatches are 0, mean is percentage of correct predictions.
    # Cast from bool to numeric value is necessary to compute mean, use float32 to avoid rounding errors
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Learning process: for num_epochs, run data in num_batches many batches and minimize cost_function
    for epoch in range( self.num_epochs ):
        total_loss = 0
        num_batches = int(train_x.shape[0] / batch_size)
        self.training_epochs.append(epoch)
        
        # for every batch, run optimizer, total loss for that batch
        for i in range(num_batches):
            batch_xs, batch_ys = train_data_feeder.next_batch(batch_size)  # get next input- and labels batch
            
            # Optimize model with the current batch
            self.sess.run(optimizer, feed_dict={x: batch_xs, y_labels: batch_ys})  

            # calculate loss of that batch, just as a reference
            total_loss += self.sess.run(cost_function, feed_dict={x: batch_xs, y_labels: batch_ys})

        # print the accuracy on training and validation set after each training epoch
        # and store it in an array
        self.accurancy_train_set.append(self.sess.run(accuracy, feed_dict={x: train_x, y_labels: train_y}))
        self.accurancy_valid_set.append(self.sess.run(accuracy, feed_dict={x: test_x, y_labels: test_y}))

        # print average loss over all batches in that epoch, as a reference
        print("Epoch {} complete, loss={}".format(epoch + 1, total_loss/num_batches))
        
    # Test the final model with an unseen testing sample
    self.final_accuracy = self.sess.run(accuracy, feed_dict={x: valid_x, y_labels: valid_y})

  ##
  #
  # Writes data to a csv file
  #
  ##
  def __write_to_file( self ):
    result_file = open("results/results-train.csv", "ab")
    writer = csv.writer(result_file, delimiter=",")
    writer.writerow( self.accurancy_train_set )
    result_file.close()

    result_file = open("results/results-test.csv", "ab")
    writer = csv.writer(result_file, delimiter=",")
    writer.writerow( self.accurancy_valid_set )
    result_file.close()

  ##
  #
  # Plots results
  # We always write the same plot so all lines should be on one plot at the end
  #
  ##
  def __plot( self ):
    plt.plot(self.training_epochs, self.accurancy_train_set, label='accuracy for train set'.format(i=1))
    plt.plot(self.training_epochs, self.accurancy_valid_set, label='accuracy for validation set (fold='+str(self.leave_out_fold)+')'.format(i=2))
    plt.legend(loc='best')
    plt.xlabel('training epoch number')
    plt.suptitle('Plot with learning rate ' + str( self.learning_rate ) + " and " + str(self.size_hidden_layer) + " hidden layers.")
    print("Plot with error-rate of training and validation set depending on training epoch is made (for cross-validation).")
    plt.savefig('plots/folds/Plot with learning rate ' + str( self.learning_rate ) + " and " + str(self.size_hidden_layer) + " hidden layers fold.")
    
learning_rates = [ 0.1 , 0.05,  0.01 ]
hidden_layers_sizes = [ 20 , 40 , 60 , 80 ]

for learning_rate in learning_rates:
  for size_hidden_layer in hidden_layers_sizes:
    # Clean results folder
    folder = 'results'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
              os.unlink(file_path)
        except Exception as e:
            print(e)
  
    plt.clf() # clear plot
    # Cross-validation
    for leave_out_fold in range(0, num_folds):
      train = Train(num_epochs, learning_rate, size_hidden_layer, leave_out_fold)
      train.start()

    # Plot full result
    plt.clf() # clear plot
    PlotFull( num_epochs , learning_rate , size_hidden_layer , num_folds )

