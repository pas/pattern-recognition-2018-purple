import numpy as np

class Features:

  def generateFV(self, x1, x2):
    featureVector = list()

    featureVector.append(self.lowerContour(x1))
    featureVector.append(self.upperContour(x1))
    featureVector.append(self.bwTransitions(x1))
    featureVector.append(self.blackPxFractionWindow(x1))
    featureVector.append(self.blackPxFractionLcUc(x1))
    featureVector.append(self.gradient(x1, x2))

    return np.asarray(featureVector)

  # The lower Contour is the black pixel on the lowest row
  def lowerContour(self, x):
    return np.where(x == x.min())[0][-1]

  # The upper Contour is the black pixel on the highest row
  def upperContour(self, x):
    return np.where(x == x.min())[0][0]

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

  def blackPxFractionLcUc(self, x):
    # The lowest Contour should also include itself for the number of black pixels
    # Therefore 1 has to be added.
    xLc = self.lowerContour(x) + 1
    xUc = self.upperContour(x)
    blackPxs = len(np.where(x[xUc:xLc,] == x.min())[0])
    return blackPxs/len(x)

  def gradient(self, x1, x2):
    x1Lc = self.lowerContour(x1)
    x1Uc = self.upperContour(x1)
    x2Lc = self.lowerContour(x2)
    x2Uc = self.upperContour(x2)        
    return x1Lc/x2Lc , x1Uc/x2Uc


""" x = [0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0]
y = [0, 0, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0]
f = Features()
test = f.generateFV(np.asarray(x), np.asarray(y))
print(test) """