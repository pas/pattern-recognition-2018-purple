from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths

paths = Paths()
results = paths.get( "data/ground-truth/locations/270.svg" )

imageP = ImageProcessor()

number = 1
for path in results:
  bounds = imageP.minmax( path )
  print( bounds )
  print( path )
  print( path.astype( "uint8" ) )
  imageP.crop( "data/images/270.jpg" , path , "images/image-"+str(number) )
  number += 1

