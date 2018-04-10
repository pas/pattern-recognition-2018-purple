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
    
  def minmax( self , arr ):
    result = []
    
    xvalues = arr[:, 0]
    yvalues = arr[:, 1]
    
    xmax = numpy.max( arr[:, 0] )
    xmin = numpy.min( arr[:, 0] )
    
    ymax = numpy.max( arr[:, 1] )
    ymin = numpy.min( arr[:, 1] ) 
    
    return numpy.array( [ xmin , xmax , ymin , ymax ] )
    
  # 
  # Gets the quadratic bounding box of a shape
  #
  def bounding_box( self, arr ):
    result = []
    
    minmax = self.minmax( arr )
    
    result.append( [ minmax[0] , minmax[2] ] )
    result.append( [ minmax[1] , minmax[2] ] )
    result.append( [ minmax[1] , minmax[3] ] )
    result.append( [ minmax[0] , minmax[3] ] )
    
    return numpy.array( result )
    
  #
  # Crops image to the given path
  #
  # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil#14211878
  #
  def crop( self , image_path , crop_path ,cropped_image_name):
    image=Image.open( image_path )
    image.load()
    image_data = numpy.asarray(image)
    
    # minX, maxX, minY, maxY
    image_data_new = image_data[ int(crop_path[0]):int(crop_path[1]), int(crop_path[2]):int(crop_path[3])]

    new_image = Image.fromarray(image_data_new)
    new_image.save(cropped_image_name+'.png')
    return 0
  
