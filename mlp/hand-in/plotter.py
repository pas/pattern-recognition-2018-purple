import matplotlib.pyplot as plt
import numpy as numpy
import csv as csv

##
# Plotting full results
# and writing best results into
# results.txt
##
class Plot:    
  def plot_full( self , num_epochs , learning_rate , size_hidden_layer , num_folds ):
    plt.clf() # reset plot
    
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
    # learning rate , number of hidden layers , number of epochs, best value , standard deviation
    to_write = [ learning_rate, size_hidden_layer, num_epochs, numpy.max(mean_test), stdev_test[ numpy.argmax(mean_test) ] ]
    writer.writerow( to_write )
    result_file.close()
    
    plt.clf()

  ##
  #
  # Plots folds results
  # We always write the same plot so all lines should be on one plot at the end.
  # This has to be done by hand outside of this method
  #
  ##
  def plot_fold( self, training_epochs , accuracy_train_set , accuracy_test_set , fold , learning_rate , size_hidden_layer ):
    training_epochs = numpy.arange(1, training_epochs+1) # create array [1, 2, ... , n] where n is the training epochs
    plt.plot( training_epochs , accuracy_train_set , label='accuracy for train set'.format(i=1))
    plt.plot( training_epochs , accuracy_test_set , label='accuracy for validation set (fold='+str(fold)+')'.format(i=2))
    plt.legend(loc='best')
    plt.xlabel('training epoch number')
    plt.suptitle('Plot with learning rate ' + str( learning_rate ) + " and " + str( size_hidden_layer ) + " hidden layers.")
    plt.savefig('plots/folds/Plot with learning rate ' + str( learning_rate ) + " and " + str( size_hidden_layer ) + " hidden layers fold.")
  
