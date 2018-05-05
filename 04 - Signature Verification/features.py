##
#
# Calculates the feature vectors.
#
##

from math import sqrt

class Features:
    def __init__(self):
        return

    # Generates a dictionary. The key is the name of the signature txt file. The value is the feature vector of the signature.
    def generate_features(self, raw_data):
        features = {}
        for k, v in raw_data.items():
            label = k # Name of the signature txt file.
            feature_vector = self.calculate_feature_vector(v)
            features[label] = feature_vector
        return features

    # Returns vector with feature array for each timestamp.
    def calculate_feature_vector(self, data):
        feature_vector = []

        distance = float('0') # Stores already made distance. Needed to calculate the speed.
        previous_point = [ float(data[0][1]), float(data[0][2]) ] # Starting point [x,y] of signature. Needed to calculate already made distance.

        for time_slice in data: # Iterate through the timepoints.
            actual_point = [ float(time_slice[1]), float(time_slice[2]) ]
            distance = distance + self.calculate_distance(previous_point, actual_point) # Calculates the distance made so far.
            feature_vector.append( self.feature_values_at_timepoint( distance, time_slice))

            previous_point = actual_point

        return feature_vector

    def calculate_distance(self, previous_point, actual_point):
        return float (sqrt ( ( previous_point[0] - actual_point[0] )**2 + (previous_point[1] - actual_point[1])**2))

    def feature_values_at_timepoint(self, distance, time_slice):
        feature_values = []
        feature_values.append(float(time_slice[1])) # x
        feature_values.append(float(time_slice[2])) # y
        feature_values.append(float(time_slice[5])) # pressure
        feature_values.append(float('0') if  float(time_slice[0])==float('0') else (distance / float(time_slice[0]))) # speed
        return feature_values
