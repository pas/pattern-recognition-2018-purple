import numpy as numpy
import Image
import cv2 as cv

##
#
# Processing images (especially cutting)
#
##

class ImageProcessor:
  
  def __init__( self , image_name ):
    self.image_name = image_name
    im = Image.open( image_name )
    im.load()
    self.image = numpy.asarray(im)
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
    
    xmax = numpy.max( arr[:, 1] )
    xmin = numpy.min( arr[:, 1] )
    
    ymax = numpy.max( arr[:, 0] )
    ymin = numpy.min( arr[:, 0] ) 
    
    return numpy.array( [ xmin , xmax , ymin , ymax ] )
  
  #
  # Creates a mask for the path
  # with the size of the image
  #
  # Be aware that the given array
  # has to uphold the strange cv array in array
  # format:
  #
  # [  
  #   [ [ x1 , y1 ] ], 
  #   [ [ x2 , y2 ] ] 
  # ]
  #
  # If you have an array [ [ x1 , y1 ] , [ x2 , y2 ] ]
  # then you can use #to_weired_format( array ) to
  # get the needed format.
  # 
  def create_mask( self , arr ):
    mask = numpy.zeros(self.image.shape)

    # First -1 to take all given shapes
    # Second -1 to fill contours
    cv.drawContours( mask , [arr] , -1 , (255, 255, 255, 255), -1 )
    
    # Everything that is white is True
    return numpy.equal(mask, 255, None)
  
  #
  # Save an image from an array to disk
  #
  def _save( self , image_data , image_name ):
    new_image = Image.fromarray(image_data)
    new_image.save(image_name+'.png')
    
  #
  # Save image as png to disk
  #
  def save( self , image_name ):
    new_image = Image.fromarray(self.image)
    new_image.save(image_name+'.png')

  # 
  # Rectangular bounding box of a shape
  #
  def bounding_box( self, shape ):
    result = []
    
    minmax = self.minmax( shape )
    
    result.append( [ minmax[0] , minmax[2] ] )
    result.append( [ minmax[1] , minmax[2] ] )
    result.append( [ minmax[1] , minmax[3] ] )
    result.append( [ minmax[0] , minmax[3] ] )
    
    return numpy.array( result )
    
  #
  # Creates format used by cv
  #
  # Example:
  #
  # image = Image("test.png")
  # image.to_weired_format( [ [ 1 , 2 ] , [ 3 , 4 ] ] )
  # # [ [ [ 1, 2 ] ] ], [ [ 3 , 4 ] ] ]
  def to_weired_format( self , points ):
    new_format = []
    
    for point in points:
      new_format.append( [ point ] )
    
    return numpy.array( new_format ).astype( int )
  #
  # Crops image to the given path
  #
  # Returns new image as numpy array
  # https://stackoverflow.com/questions/26049174/creating-a-boolean-array-which-compares-numpy-elements-to-none
  # https://stackoverflow.com/questions/40916035/drawcontours-fails-to-create-a-correct-mask-in-opencv-3-1-0#40917870
  # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil#14211878
  #
  def crop( self , crop_path ):
    # Create mask    
    mask = self.create_mask( self.to_weired_format( crop_path ) )
    
    # Create empty image
    res = numpy.zeros(self.image.shape)
    res[res == 0] = 255
    
    # Copy only parts in mask to new empty image
    numpy.copyto(res, self.image, where=mask)
    
    # Get boundaries
    boundaries = self.minmax( crop_path )
    
    # Cut image to boundaries
    image_data_new = res[ int(boundaries[0]):int(boundaries[1]), int(boundaries[2]):int(boundaries[3])]

    return image_data_new
  
