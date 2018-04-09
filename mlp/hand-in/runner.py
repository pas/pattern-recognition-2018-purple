##
#
# Runs a full cicle with all the learning rates
# and all the possible hidden layers.
#
# Run this with 
# run.py -e 100 -f 4
#
##

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as numpy
import sys
import getopt
import csv as csv
import os as os
import matplotlib.pyplot as plt

from data_loader import load_data
from batch_feed import RandomBatchFeeder

# Created by Pas
from train import Train
from loader import Loader
from plotter import Plot

try:
  opts, args = getopt.getopt(sys.argv[1:], "he:f:", ["help","num_epochs=","num_folds="])
except getopt.GetoptError:
  print getopt.GetoptError.msg
  print 'runner.py -e <number of Epochs> -f <number of Folds>'
  sys.exit(2)

for opt, arg in opts:
  if opt in ("-h", "--help"):
    print 'runner.py -e <number of Epochs> -f <number of Folds>'
    sys.exit()
  elif opt in ("-e", "--num_epochs"):
    num_epochs = int(arg)
  elif opt in ("-f", "--num_folds"):
    num_folds = int(arg)
    
# Learning rates to test
learning_rates = [ 0.009 ]

# Hidden layer sizes to test
hidden_layers_sizes = [ 20 , 40 , 60 , 80 ]

# Used to load training and test data
loader = Loader()
plot = Plot()

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
  
    plt.clf()
    # Cross-validation
    for leave_out_fold in range(0, num_folds):
      train = Train(num_epochs, learning_rate, size_hidden_layer, leave_out_fold, num_folds)
      (train_x, train_y, test_x, test_y) = loader.from_folds( num_folds , leave_out_fold )
      train.start(train_x, train_y, test_x, test_y)
      plot.plot_fold( num_epochs , train.get_accuracy_train_set() , train.get_accuracy_test_set() , leave_out_fold , learning_rate , size_hidden_layer )

    # Plot full result
    plot.plot_full( num_epochs , learning_rate , size_hidden_layer , num_folds )

