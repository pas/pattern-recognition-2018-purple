import numpy as numpy
import sys

from data_loader import load_data

class Loader:
  # This part expects that there are files in the folds folder. It combines all files except
  # the left out as train set and returns the left out as test set

  def from_folds( self , num_folds , leave_out_fold):
    train_x = numpy.array([])
    train_y = numpy.array([])
    empty = True

    # Create train and test data from folds
    for fold in range( 0 , num_folds ):
      fold_file = "folds/fold-"+str(fold)+".csv"
      
      if( fold != leave_out_fold ):
        current_x, current_y = load_data( fold_file )
        
        # Too lazy to find a better solution to make this look more slick
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

    return (train_x, train_y, test_x, test_y)
    
  def from_files( self , train_file_str , test_file_str ):
    train_x, train_y = load_data( train_file )
    test_x, test_y = load_data( test_file )
    return (train_x, train_y, test_x, test_y)
