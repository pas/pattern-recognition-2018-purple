import numpy as numpy
import math


##
#
# Copied from assignment 3
#
# Currently more or less from
# https://en.wikipedia.org/wiki/Dynamic_time_warping
#
##

class DTW:
    def distance(self, s, t, w):
        s = numpy.array(s)
        t = numpy.array(t)

        n = len(s)
        m = len(t)

        res = numpy.empty((n, m), dtype=numpy.dtype('float'))
        res.shape = (n, m)
        res.fill(float('inf'))

        w = max(w, abs(n - m))  # adapt window size (*)

        for i in range(0, n):
            for j in range(0, m):
                res[i][j] = float('inf')

        res[0][0] = 0

        for i in range(0, n):
            for j in range(max(1, i - w), min(m, i + w)):
                if ((i, j) != (0, 0)):
                    cost = self.feature_distance(s[i] - t[j])
                    res[i][j] = cost + min(res[i - 1][j],  # insertion
                                           res[i][j - 1],  # deletion
                                           res[i - 1][j - 1])  # substitution

        return res[n - 1][m - 1], res

    def feature_distance(self, value_array):
        return math.sqrt(value_array[0]**2 + value_array[1]**2 + value_array[2]**2 + value_array[3]**2)





