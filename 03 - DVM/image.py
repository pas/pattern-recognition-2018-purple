import numpy as numpy
import Image

##
#
# Processing images (especially cutting)
#
##

class ImageProcessor:
  #
  # Resizes the image to the given
  # height or width
  #
  def resize( image , height , width ):
    # ...
    
    return 0
    
  # 
  # Gets the quadratic bounding box of a shape
  #
  def boundingBox( shape ):
    # Get maxX, maxY, minX, minY
    # Return [ upperLeft : [ minX, minY], lowerLeft : [ maxX, minY ] upperRight : [ minX , maxY ] lowerRight : [ maxX, maxY ] ]
    
    return 0
    
  #
  # Crops image to the given path
  #
  # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil#14211878
  #
  def crop( self , image_path , crop_path ):
    image=Image.open( image_path )
    image.load()
    image_data = numpy.asarray(image)
    
    print(image_data)
    #boundingBox( path )
    image_data_new = image_data[crop_path[0]:crop_path[1], crop_path[2]:crop_path[3], :]

    new_image = Image.fromarray(image_data_new)
    new_image.save(image_path+'_cropped.png')
    return 0
  
