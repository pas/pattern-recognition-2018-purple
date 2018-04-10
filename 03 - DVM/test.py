import numpy as numpy

from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths

import unittest

class Tests(unittest.TestCase):
  
  #
  # ImageProcessor
  #
  def test_image_processor( self ):
    imageP = ImageProcessor()
    
    shape = numpy.array( [ [ 45 , 45 ] , [ 75 , 155 ] ,  [ 155 , 45 ] ] )
    bounds = imageP.bounding_box( shape )
    
    numpy.testing.assert_array_equal( bounds , numpy.array( [ [ 45.0 , 45.0 ] , [ 155.0 , 45.0] , [155.0 , 155.0] , [45.0 , 155.0] ] ) )

    imageP.crop( "test.png" , imageP.minmax( shape ) , "test_cropped.png" )

    # imageP.resize( ... )
  
  #
  # Preprocessor
  #
  def test_pre_processor( self ):
    preprocess = Preprocessor()

    res = preprocess.normalize( numpy.array( [ 3.0 , 4.0 , 2.0 , 1.0 ] ) , 1 , 5 )
    self.assertTrue( ( numpy.asarray( res ) == [ 0.5 , 0.75 , 0.25 , 0 ] ).all() )

    res = preprocess.get_black_and_white_ratio( [ 0 , 1 , 0 , 0 ] )
    self.assertEqual( res , 0.25 )

  #
  # Paths (XML - SVG)
  #
  def test_paths( self ):
    paths = Paths()
    results = paths.get( "test.svg" )
    numpy.testing.assert_array_equal( results[0] , numpy.array( [ [ 45.0 , 45.0 ] , [ 45.0 , 155.0] , [155.0 , 155.0] , [155.0 , 45.0] ] ) )

if __name__ == '__main__':
    unittest.main()
