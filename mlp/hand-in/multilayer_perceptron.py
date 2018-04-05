import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as numpy
import sys
import getopt
import csv as csv
import os as os

from data_loader import load_data
from batch_feed import RandomBatchFeeder

# Created by Pas
from train import Train
from loader import Loader
from plotter import PlotFull

try:
  opts, args = getopt.getopt(sys.argv[1:], "he:f:", ["help","num_epochs=","num_folds="])
except getopt.GetoptError:
  print getopt.GetoptError.msg
  print 'multilayer_perceptron.py -e <number of Epochs> -f <Number of folds>'
  sys.exit(2)

for opt, arg in opts:
  if opt in ("-h", "--help"):
    print 'multilayer_perceptron.py -e <number of Epochs> -f <Fold number>'
    sys.exit()
  elif opt in ("-e", "--num_epochs"):
    num_epochs = int(arg)
  elif opt in ("-f", "--num_folds"):
    num_folds = int(arg)
    
learning_rates = [ 0.1 ]
hidden_layers_sizes = [ 20 ]

# Used to load training and test data
loader = Loader()

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
  
    # Cross-validation
    for leave_out_fold in range(0, num_folds):
      train = Train(num_epochs, learning_rate, size_hidden_layer, leave_out_fold, num_folds)
      (train_x, train_y, test_x, test_y) = loader.from_folds( num_folds , leave_out_fold )
      train.start(train_x, train_y, test_x, test_y)

    # Plot full result
    PlotFull( num_epochs , learning_rate , size_hidden_layer , num_folds )

