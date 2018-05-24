import numpy as numpy

from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths
from dtw import DTW
from features import Features
from metrics import Metrics

import Image

import unittest

class Tests(unittest.TestCase):
  
  #
  # Metrics
  # 
  def test_metrics( self ):
    # Full sample: 15
    # Selected 11 ( 5 true positives , 6 false positives )
    # Not-selected 4 ( 3 true negatives , 1 false negatives )
    # 8 correctly selected ( 5 true positives, 3 true negatives)
    
    # tp , tn , fp , fn
    metrics = Metrics( 5 , 3 , 6 , 1 )
    
    # tp / (tp + fn) => 5 / ( 5 + 1 )
    self.assertEqual( metrics.recall() , 5/6 )
    
    # tp / (tp + fp) => 5 / ( 5 + 6 )
    self.assertEqual( metrics.precision() , 5/11 )
    
    metrics = Metrics( 1 , 2 , 0 , 2 )
    self.assertEqual( metrics.precision() , 1 )
    
    res = [ True , True , False , True , False ]
    recall, precision = Metrics.plot_recall_precision( res , "any" , "any.png" )
    
    numpy.testing.assert_array_equal( recall , [ 1/3 , 2/3 , 2/3 , 1 , 1 ] )
    numpy.testing.assert_array_equal( precision , [ 1 , 1 , 2/3 , 3/4 , 3/5 ] )
    
  
  #
  # ImageProcessor
  #
  def test_image_processor( self ):
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
  # Features
  #
  
  def test_features( self ):
    features = Features()
    
    res = features.lowerContour( numpy.array( [ 255 , 0 , 255 , 0 , 0 ] ) )
    self.assertEqual( res , 4 )
    res = features.lowerContour( numpy.array( [ 255 , 255 , 255 , 255 , 255 ] ) )
    self.assertEqual( res , 4 )
    res = features.upperContour( numpy.array( [ 255 , 0 , 255 , 0 , 0 ] ) )
    self.assertEqual( res , 1 )
    res = features.upperContour( numpy.array( [ 255 , 255 , 255 , 255 , 255 ] ) )
    self.assertEqual( res , 0 )
    
    
    res = features.blackPxFractionWindow( numpy.array( [ 0 , 0 , 0 , 0 , 0 ] ) )
    self.assertEqual( res , 1 )
    
    res = features.blackPxFractionWindow( numpy.array( [ 0 , 255 , 0 , 255 , 255 ] ) )
    self.assertEqual( res , 0.4 )
    
    res = features.bwTransitions( numpy.array( [ 0 , 255 , 0 , 255 , 0 ] ) )
    self.assertEqual( res , 5 )
    
    res = features.bwTransitions( numpy.array( [ 255 , 255 , 0 , 255 , 255 ] ) )
    self.assertEqual( res , 2 )
    
    res = features.bwTransitions( numpy.array( [ 0 , 0 , 0 , 0 , 0 ] ) )
    self.assertEqual( res , 1 )
    
    res = features.bwTransitions( numpy.array( [ 255 , 255 , 255 , 255 , 255 ] ) )
    self.assertEqual( res , 0 )
    
    #res = features.blackPxFractionWindow( numpy.array( [ 255 , 255 , 255 , 255 , 255 ] ) )
    #self.assertEqual( res , 0 )
    
    res = features.blackPxFractionLcUc( numpy.array( [ 255 , 0 , 255 , 0 , 0 ] ) )
    self.assertEqual( res , 0.75 )
    
    res1 , res2 = features.gradient( numpy.array( [ 0 , 0 , 0 , 0 , 0 ] ) , numpy.array( [ 0 , 0 , 0 , 0 , 0 ] ) )
    self.assertEqual( res1 , 0 )
    self.assertEqual( res2 , 0 )
    
    res1 , res2 = features.gradient( numpy.array( [ 255 , 0 , 0 , 0 , 0 ] ) , numpy.array( [ 0 , 255 , 0 , 0 , 0 ] ) )
    self.assertEqual( res1 , 0 )
    self.assertEqual( res2 , 1 )
    
  #
  # DTW
  #
  
  def test_dtw( self ):
    dtw = DTW()
    dist, _ = dtw.distance( numpy.array([ 1 , 2 , 3 ]) , numpy.array( [1 , 1 , 2 , 2 , 3 , 3 ] ) , 100 )
    self.assertEqual( dist , 0.0 )
    
    dist = dtw.distance( numpy.array([ 1 , 1 , 3 ]) , numpy.array( [1 , 1 , 2 , 2 , 3 , 3 ] ) , 100 )
    dist = dtw.distance( numpy.array([ 1 , 2 , 1 , 2 , 2 , 2 ]) , numpy.array( [ 2 , 1 , 2 , 1 , 1 , 1 ] ) , 2 )
    #print( dist )
    
    preprocess = Preprocessor()
    image1 = ImageProcessor( "images/270/image-22.png" ) #to
    features1 = image1.calculate_feature_vectors( )
    
    image2 = ImageProcessor( "images/270/image-32.png" ) #to
    features2 = image2.calculate_feature_vectors()
    
    dist, _ = dtw.distance( numpy.array(features1) , numpy.array(features2) , len(features1) + len(features2) )
    print( dist )
      
    # train
    image3 = ImageProcessor( "binarization-test.png" ) # image-27.img = of
    for line in image3.image :
        print ( line )

    print( image3.image )
    features3 = image3.calculate_feature_vectors()
    
    # find
    '''
    for image_number in range( 1 , 221 ):  
      image = ImageProcessor( "images/270/image-"+str(image_number)+".png" )
      
      features_current = image.calculate_feature_vectors()
      dist, _ = dtw.distance( numpy.array( features3 ) , numpy.array( features_current ) , len( features3 ) + len( features_current ) )
      if( dist < 2000 ):
        print( dist )
        print( "270/image-"+str(image_number) )
    '''
    
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
