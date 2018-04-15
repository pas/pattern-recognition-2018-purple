import numpy as numpy

##
#
# Implementing the
# dynamic time warping
#
# Currently more or less from
# https://en.wikipedia.org/wiki/Dynamic_time_warping
#
##

class DTW:
  def distance(self, s, t , w ):
    n = len(s)
    m = len(t)
    
    res = numpy.empty( ( n, m ) , dtype=numpy.dtype('float') )
    res.shape = ( n , m )
    res[ res == 0 ] = float('inf')

    w = max( w , abs(n-m) ) # adapt window size (*)

    for i in range(0, n):
      for j in  range(0, m):
        res[i][j] = float('inf')
        
    res[0][0] = 0

    for i in range( 0 , n ):
      for j in  range( max( 1 ,  i-w ) , min( m , i+w ) ):
        if( (i,j) != (0,0) ): 
          cost = self.feature_distance( s[i] , t[j] )
          res[i][j] = cost + min( res[ i-1 ][ j   ],  #insertion
                                  res[ i   ][ j-1 ],  #deletion
                                  res[ i-1 ][ j-1 ] ) #substitution

    return res[n-1][m-1], res
  
  def feature_distance( self , value1 , value2 ):
    return numpy.linalg.norm( value1 - value2 )

  


