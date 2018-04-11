import numpy as numpy
import cv2 as cv

##
#
# Methods to preprocess the data
#
##

class Preprocessor: 
  #
  # Creates a "binarized" an image
  # matrix
  #
  # Expects values from 0 to 255
  def binarization( self , image ):
    ret, res = cv.threshold( image , 127 , 255 , cv.THRESH_BINARY )
    return res
    
  #
  # Normalizes data in a vector
  #
  # Example:
  # normalize( [ 3, 4 , 2 , 1 ] , 1 , 5 )
  # returns [ 0.75 , 1 , 0.5 , 0.25 ]
  #
  def normalize( self , vector , mini , maxi ):
    return (vector - mini) / (maxi - mini)
  
  #
  # Gets the number of black and white
  # transition in a image vector
  #
  # Expects a row vector
  #
  # Example:
  # get_nr_of_transitions( [ 0 , 0 , 1 , 0 , 1 , 0 , 0 , 1 ]
  # returns 5
  #
  def get_nr_of_transitions( vector ):
    # iterate through vector and count transitions
    return vector
    
  #
  # Gets the ratio of white
  # pixel in a "binarized" image
  # vector
  #
  # Expects a row vector. Interprets
  # 1 as white and 0 as black.
  #
  # Example:
  # get_black_and_white_ratio( [ 0 , 1 , 0 , 0 ] )
  # returns 0.25 ( 1/4 )
  #
  def get_black_and_white_ratio( self , vector ):
    print( len(vector) )
    # get size of vector
    # get number of 1 in vector
    # calculate 1s / size
    return vector.count( 1 ) / float( len( vector ) )
  
  
    
    
  
  
