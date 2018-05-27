#
# Extracting paths from svg
#
# https://docs.python.org/3/library/xml.etree.elementtree.html
#

import xml.etree.ElementTree as ET
import numpy as numpy


class Paths:
  #
  # Get all paths in the svg file
  # as array in array
  #
  # Second argument is a array of all
  # of the paths
  #
  # Example:
  # [ numpy.array([ [0, 155] , [100, 200] ]), numpy.array( ... ) , .. ]
  # [ id1, id2, ... ]
  
  
  def get( self , svg_path ):
    tree = ET.parse( svg_path )
    root = tree.getroot()
    
    all_paths = []
    ids = []

    # Expecting that svg is only holding paths
    for path in ( root.iter() ):
      id = path.get('id')
      path = path.get('d')
      if( path != None ):
        path = path.replace("M","")
        path = path.replace("L","")
        path = path.replace("Z","")
        paths = path.split()
        
        # Make bundles with x and y coordinate
        path_arr = []

        for index in range( 0 , int( len(paths)/2 ) ) :
          path_arr.append( [ float(paths[index*2]), float(paths[index*2+1]) ] )
        
        all_paths.append( numpy.array( path_arr ) )
        ids.append( id )
        
    return all_paths, ids
          
        
          
          
