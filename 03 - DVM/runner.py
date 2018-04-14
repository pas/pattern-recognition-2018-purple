import Image

from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths

paths = Paths()
results = paths.get( "data/ground-truth/locations/270.svg" )

imageP = ImageProcessor( "data/images/270.jpg" )

number = 1

for path in results:  
  new_image = imageP.crop( path )
  new_image = Image.fromarray( new_image.astype( "uint8" ) )
  new_image.save("images/image-"+str(number) +'.png')
  
  number += 1

