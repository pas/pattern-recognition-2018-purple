import numpy as numpy

from image import ImageProcessor 
from preprocess import Preprocessor 

#
# ImageProcessor
#

imageP = ImageProcessor()

# imageP.boundingBox( ... )
# imageP.resize( ... )

preprocess = Preprocessor()

res = preprocess.normalize( numpy.array( [ 3.0 , 4.0 , 2.0 , 1.0 ] ) , 1 , 5 )
print( res )

res = preprocess.get_black_and_white_ratio( [ 0 , 1 , 0 , 0 ] )
print ( res )

imageP.crop( "test.png" , [ 45 , 155 , 45 , 155 ] )
