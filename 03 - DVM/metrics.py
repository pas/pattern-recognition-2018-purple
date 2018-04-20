import numpy

##
#
# Provides function for different metrics
# to evaluate the data (f.e. recall, precision)
#
##

import matplotlib.pyplot as plt

class Metrics:
  def __init__( self , true_positives , true_negatives , false_positives , false_negatives ):
    self.true_positives = true_positives
    self.true_negatives = true_negatives
    self.false_positives = false_positives
    self.false_negatives = false_negatives
  
  def recall( self ):
    return self.true_positives / ( self.true_positives + self.false_negatives )
  
  def precision( self ):
    return self.true_positives / ( self.true_positives + self.false_positives )

  @staticmethod
  def plot_recall_precision( values ):
    precision = []
    recall = []
    
    for value in range(1, len(values)+1): 
      selected = numpy.array( values[0:value] )
      nselected = numpy.array( values[value:len(values)] )
      
      met = Metrics( ( selected == True ).sum() , ( nselected == False ).sum() , ( selected == False ).sum() , ( nselected == True ).sum()  )
      precision.append( met.precision() )
      recall.append( met.recall() )
    
    plt.plot(precision, recall, 'ro')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    plt.title('Precision-Recall curve')
    plt.savefig('pre-rec-curve.png')
    
    return recall, precision
