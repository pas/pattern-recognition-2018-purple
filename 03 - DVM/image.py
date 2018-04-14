import numpy as numpy
import Image
import cv2 as cv

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
    
    xmax = numpy.max( arr[:, 1] )
    xmin = numpy.min( arr[:, 1] )
    
    ymax = numpy.max( arr[:, 0] )
    ymin = numpy.min( arr[:, 0] ) 
    
    return numpy.array( [ xmin , xmax , ymin , ymax ] )
  
  #
  # Creates a mask
  # Be aware that the given array
  # has to uphold the strange cv array in array
  # arrangement:
  #
  # [  
  #   [ [ x1 , y1 ] ], 
  #   [ [ x2 , y2 ] ] 
  # ]
  # 
  def create_mask( self , image_path , arr ):
    image=Image.open( image_path )
    image.load()
    image_data = numpy.asarray(image)
    
    mask = numpy.zeros(image_data.shape)

    # First -1 to take all given shapes
    # Second -1 to fill contours
    cv.drawContours( mask , [arr] , -1 , (255, 255, 255, 255), -1 )
    
    # Everything that is white is True
    return numpy.equal(mask, 255, None)
  
  #
  # Save an image from an array to disk
  #
  def save( self , image_data , image_name ):
    new_image = Image.fromarray(image_data)
    new_image.save(image_name+'.png')

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
    
  def to_weired_format( self , points ):
    new_format = []
    
    for point in points:
      new_format.append( [ point ] )
    
    return numpy.array( new_format ).astype( int )
  #
  # Crops image to the given path
  #
  # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil#14211878
  #
  def crop( self , image_path , crop_path ,cropped_image_name):
    image=Image.open( image_path )
    image.load()
    image_data = numpy.asarray(image)
    
    print( self.to_weired_format( crop_path ) )
    mask = self.create_mask( image_path , self.to_weired_format( crop_path ) )
    res = numpy.zeros(image_data.shape)
    res[res == 0] = 255
    numpy.copyto(res, image_data, where=mask)
    
    crop_path = self.minmax( crop_path )
    print( crop_path )
    # minX, maxX, minY, maxY
    image_data_new = res[ int(crop_path[0]):int(crop_path[1]), int(crop_path[2]):int(crop_path[3])]

    new_image = Image.fromarray( image_data_new.astype( "uint8" ) )
    new_image.save(cropped_image_name+'.png')
    return 0
  
