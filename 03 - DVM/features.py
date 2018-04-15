import numpy as np

class Features:

  def generateFV(self, x1, x2):
    featureVector = list()

    featureVector.append(self.lowerContour(x1))
    featureVector.append(self.upperContour(x1))
    featureVector.append(self.bwTransitions(x1))
    featureVector.append(self.blackPxFractionWindow(x1))
    featureVector.append(self.blackPxFractionLcUc(x1))
    featureVector.append(self.gradientLC(x1, x2))
    featureVector.append(self.gradientUC(x1, x2))

    return np.asarray(featureVector)

  # The lower Contour is the black pixel on the lowest row
  def lowerContour(self, x):
    blackPxs = np.where(x == 0)[0]
    # When there aren't any black pixels, the lowest position of x is returned.
    if not blackPxs.any():
      return 0
    else:
      return blackPxs[-1]

  # The upper Contour is the black pixel on the highest row
  def upperContour(self, x):
    blackPxs = np.where(x == 0)[0]
    if not blackPxs.any():
      return len(x)-1
    else:
      return blackPxs[0]

  def bwTransitions(self, x):
    transitions = 0
    tmp = x[0]
    for val in x:
      if (tmp != val):
        transitions += 1
      tmp = val
    return transitions

  def blackPxFractionWindow(self, x):
    # Amount of black pixels in the array
    blackPxs = np.where(x == 0)[0]
    if not blackPxs.any():
      return 0
    try:
      result = len(blackPxs)/len(x)
      return result
    except ZeroDivisionError:
      return 0

  def blackPxFractionLcUc(self, x):
    # The lowest Contour should also include itself for the number of black pixels
    # Therefore 1 has to be added.
    xLc = self.lowerContour(x) + 1
    xUc = self.upperContour(x)
    blackPxs = np.where(x[xUc:xLc,] == 0)[0]
    try:
      result = len(blackPxs)/len(x)
      return result
    except ZeroDivisionError:
      return 0


  def gradientLC(self, x1, x2):
    x1Lc = self.lowerContour(x1)
    x2Lc = self.lowerContour(x2)
    return x2Lc-x1Lc

  def gradientUC(self, x1, x2):
    x1Uc = self.upperContour(x1)
    x2Uc = self.upperContour(x2)        
    return x2Uc-x1Uc

