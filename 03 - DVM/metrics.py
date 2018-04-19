##
#
# Provides function for different metrics
# to evaluate the data (f.e. recall, precision)
#
##

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
