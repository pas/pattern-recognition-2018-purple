#
# Images are stored into images
#

import Image

from image import ImageProcessor 
from preprocess import Preprocessor 
from paths import Paths

paths = Paths()
results = paths.get( "data/ground-truth/locations/270.svg" )

imageP = ImageProcessor( "data/images/270.jpg" )

preprocess = Preprocessor()

number = 1

images = []

for path in results:  
  new_image = imageP.crop( path )
  new_image = preprocess.binarization( new_image )
  images.append( new_image )
  #new_image = Image.fromarray( new_image )
  #new_image.save("images/image-"+str(number) +'.png')
  number += 1
  
print(images[1][1][1])

