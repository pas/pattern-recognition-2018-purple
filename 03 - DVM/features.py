import numpy as np

class Features:

  def generateFV(self, x1, x2):
    featureVector = []

    featureVector.append(self.lowerContour(x1))
    featureVector.append(self.upperContour(x1))
    featureVector.append(self.bwTransitions(x1))
    featureVector.append(self.blackPxFractionWindow(x1))
    featureVector.append(self.blackPxFractionLcUc(x1))
    a1, a2 = self.gradient(x1, x2)
    featureVector.append( a1  )
    featureVector.append( a2 )

    return np.array(featureVector)

  # The lower Contour is the black pixel on the lowest row
  def lowerContour(self, x):
    return np.where(x == x.min())[0][-1]

  # The upper Contour is the black pixel on the highest row
  def upperContour(self, x):
    return np.where(x == x.min())[0][0]

  # Counts the number of transitions from 
  # white to black and back
  # Each transition from and to
  # black is counted
  # An only white vector returns 0
  # An only black vector returns 1
  def bwTransitions(self, x):
    transitions = 0
    tmp = 255
    for val in x:
      if (tmp != val):
        transitions += 1
      tmp = val
    return transitions

  def blackPxFractionWindow(self, x):
    # Amount of black pixels in the array
    blackPxs = len(np.where(x == x.min())[0])
    return blackPxs/len(x)

  #
  # This calculates percentage 
  # of black pixels between 
  # the lower and the upper contour
  #
  # Currently this should return the
  # same as blackPxFractionWindow.
  # We should change this!
  #
  def blackPxFractionLcUc(self, x):
    # The lowest Contour should also include itself for the number of black pixels
    # Therefore 1 has to be added.
    xLc = self.lowerContour(x) + 1
    xUc = self.upperContour(x)
    
    fraction_length = xLc-xUc
    blackPxs = len(np.where(x[xUc:xLc,] == x.min())[0])
    return blackPxs/fraction_length

  #
  # First result: lc
  # Second result: uc
  #
  def gradient(self, x1, x2):
    x1Lc = self.lowerContour(x1)
    x1Uc = self.upperContour(x1)
    x2Lc = self.lowerContour(x2)
    x2Uc = self.upperContour(x2)        
    return x1Lc-x2Lc , x1Uc-x2Uc


""" x = [0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0]
y = [0, 0, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0]
f = Features()
test = f.generateFV(np.asarray(x), np.asarray(y))
print(test) """
