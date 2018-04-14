import numpy as numpy

from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths

import Image

import unittest

class Tests(unittest.TestCase):
  
  #
  # ImageProcessor
  #
  def test_image_processor( self ):
    imageP = ImageProcessor()
    
    shape = numpy.array( [ [ 45 , 45 ] , [ 75 , 155 ] ,  [ 155 , 45 ] ] )
    bounds = imageP.bounding_box( shape )
    
    print( imageP.to_weired_format( shape ) )
    
    numpy.testing.assert_array_equal( bounds , numpy.array( [ [ 45.0 , 45.0 ] , [ 155.0 , 45.0] , [155.0 , 155.0] , [45.0 , 155.0] ] ) )

    imageP.crop( "test.png" , shape , "test_cropped" )
    
    shape3 = numpy.array( [ [112, 170], [112, 230], [129, 231], [132, 230], [232, 230], [240, 238], [299,  148], [192, 147] ] )
    imageP.crop( "data/images/270.jpg", shape3, "original_shape_cropped" )
    
    # cv somehow expects a really weired structure. The points
    # are arrays of arrays but each array of array always only
    # holds one point.
    shape2 = numpy.array( [ [ [ 45 , 45 ] ] , [ [ 75 , 155 ] ] ,  [ [ 155 , 45 ] ] ] )
    imageP.create_mask( "test.png" , shape2 )

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
    
    image=Image.open( "binarization-test.png" )
    image.load()
    image_data = numpy.asarray(image)
    
    res = preprocess.binarization( image_data )
    
    expected = numpy.array( 
      [ [ [ 255 , 255 , 255 ] , [ 255 , 255 , 255 ] , [ 255 , 255 , 255 ] ],
        [ [ 0 , 0 , 0 ] , [ 0 , 0 , 0 ] , [ 0 , 0 , 0 ] ],
        [ [ 255 , 255 , 255 ] , [ 255 , 255 , 255 ] , [ 255 , 255 , 255 ] ] ] )
    
    numpy.testing.assert_array_equal(  expected , res )

  #
  # Paths (XML - SVG)
  #
  def test_paths( self ):
    paths = Paths()
    results = paths.get( "test.svg" )
    numpy.testing.assert_array_equal( results[0] , numpy.array( [ [ 45.0 , 45.0 ] , [ 155.0 , 45.0] , [155.0 , 155.0] , [45.0 , 155.0] ] ) )

if __name__ == '__main__':
    unittest.main()
