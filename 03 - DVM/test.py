import numpy as numpy

from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths
from dtw import DTW

import Image

import unittest

class Tests(unittest.TestCase):
  
  #
  # ImageProcessor
  #
  def test_image_processor( self ):
    print(Image.__file__)
    
    test_image = ImageProcessor( "test.png" )
    
    shape = numpy.array( [ [ 45 , 45 ] , [ 75 , 155 ] ,  [ 155 , 45 ] ] )

    # Testing to_weired_format
    # cv somehow expects a really weired structure. The points
    # are arrays of arrays but each array of array always only
    # holds one point.
    self.assertTrue( test_image.to_weired_format( shape ).shape == (3 , 1 , 2 ) )
    
    # Testing bounding_box
    bounds = test_image.bounding_box( shape )
    numpy.testing.assert_array_equal( bounds , numpy.array( [ [ 45.0 , 45.0 ] , [ 155.0 , 45.0] , [155.0 , 155.0] , [45.0 , 155.0] ] ) )

    # Testing crop (only visible test!)
    res = test_image.crop( shape )
    new_image = Image.fromarray( res )
    new_image.save("test_cropped.png")
    
    # Testing crop with original picture (only visible test!)
    original_image = ImageProcessor( "test.png" )
    
    shape3 = numpy.array( [ [112, 170], [112, 230], [129, 231], [132, 230], [232, 230], [240, 238], [299,  148], [192, 147] ] )
    res = original_image.crop( shape3 )
    new_image = Image.fromarray( res )
    new_image.save("original_shape_cropped.png")
    
    # Testing mask
    mini_test_image = ImageProcessor( "binarization-test.png" )
    quadratic_shape = numpy.array( [ [ [ 0 , 0 ] ] , [ [ 0 , 1 ] ] ,  [ [ 1 , 1 ] ] , [ [ 1 , 0 ] ] ] )
    mask = mini_test_image.create_mask( quadratic_shape )
    
    expected = numpy.array([ 
        [ [ True,  True ,  True  ],
          [ True,  True ,  True  ],
          [ False, False , False ] ],

        [ [ True,  True,  True  ],
          [ True,  True,  True  ],
          [ False, False, False ] ],

        [ [ False, False, False ],
          [ False, False, False ],
          [ False, False, False ] ] ])
    
    numpy.testing.assert_array_equal( expected , mask )

    # imageP.resize( ... )
    
  #
  # DTW
  #
  
  def test_dtw( self ):
    dtw = DTW()
    dist = dtw.distance( numpy.array([ 1 , 2 , 3 ]) , numpy.array( [1 , 1 , 2 , 2 , 3 , 3 ] ) , 100 )
    dist = dtw.distance( numpy.array([ 1 , 1 , 3 ]) , numpy.array( [1 , 1 , 2 , 2 , 3 , 3 ] ) , 100 )
    dist = dtw.distance( numpy.array([ 1 , 2 , 1 , 2 , 2 , 2 ]) , numpy.array( [ 2 , 1 , 2 , 1 , 1 , 1 ] ) , 2 )
    #print( dist )
    
    preprocess = Preprocessor()
    im = Image.open( "images/image-22.png" ) #to
    im.load()
    image1 = numpy.asarray(im)
    features1 = []
    
    for line in image1:
      features1.append( preprocess.get_black_and_white_ratio( line.tolist() , 255 ) )
    
    im = Image.open( "images/image-32.png" ) #to
    im.load()
    image2 = numpy.asarray(im)
    
    features2 = []
    for line in image2:
      features2.append( preprocess.get_black_and_white_ratio( line.tolist() , 255 ) )
    
    
    dist, _ = dtw.distance( numpy.array(features1) , numpy.array(features2) , len(features1) + len(features2) )
    print( dist )

    features3 = []
    for line in image2:
      features2.append( preprocess.get_black_and_white_ratio( line.tolist() , 255 ) )
      
    im = Image.open( "images/image-27.png" ) #of
    im.load()
    image3 = numpy.asarray(im)

    features3 = []
    for line in image3:
      features3.append( preprocess.get_black_and_white_ratio( line.tolist() , 255 ) )
    
    dist, _ = dtw.distance( numpy.array(features1) , numpy.array(features3) , len(features1) + len(features3) )
    print( dist )
    
    
  #
  # Preprocessor
  #
  def test_pre_processor( self ):
    preprocess = Preprocessor()

    res = preprocess.normalize( numpy.array( [ 3.0 , 4.0 , 2.0 , 1.0 ] ) , 1 , 5 )
    self.assertTrue( ( numpy.asarray( res ) == [ 0.5 , 0.75 , 0.25 , 0 ] ).all() )

    res = preprocess.get_black_and_white_ratio( [ 0 , 1 , 0 , 0 ] , 1 )
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
    numpy.testing.assert_array_equal( results[0] , numpy.array( [ [ 45.0 , 45.0 ] , [ 45.0 , 155.0] , [155.0 , 155.0] , [155.0 , 45.0] ] ) )

if __name__ == '__main__':
    unittest.main()
