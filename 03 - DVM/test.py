import numpy as numpy

from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths

#
# ImageProcessor
#

imageP = ImageProcessor()

imageP.crop( "test.png" , [ 45 , 155 , 45 , 155 ] )

# imageP.boundingBox( ... )
# imageP.resize( ... )

#
# Preprocessor
#

preprocess = Preprocessor()

res = preprocess.normalize( numpy.array( [ 3.0 , 4.0 , 2.0 , 1.0 ] ) , 1 , 5 )
print( res )

res = preprocess.get_black_and_white_ratio( [ 0 , 1 , 0 , 0 ] )
print ( res )

#
# Paths (XML - SVG)
#

paths = Paths()
paths.get( "test.svg" )

