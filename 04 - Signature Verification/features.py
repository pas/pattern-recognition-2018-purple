##
#
# Calculates the feature vectors.
#
##

from math import sqrt
import pandas as pd
import numpy as np

class Features:

    EXPECTED_NUM_FEATURES = 4

    def __init__(self):
        return

    # Generates a dictionary. The key is the name of the signature txt file.
    # The value is the feature vector of the signature.
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
        # Starting point [x,y] of signature. Needed to calculate already made distance.
        previous_point = [ float(data[0][1]), float(data[0][2]) ]
        

        for time_slice in data: # Iterate through the timepoints.
            actual_point = [ float(time_slice[1]), float(time_slice[2]) ]
            # Calculates the distance made so far.
            distance = distance + self.calculate_distance(previous_point, actual_point)
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
        # speed
        feature_values.append(float('0') if  float(time_slice[0])==float('0') else (distance / float(time_slice[0]))) 
        return feature_values

    def normalize_signature_features(self, features_dict):
        """
        Takes as input a dictionary with lists of feature vectors as values, returns the same dictionary with each
        signature (i.e. each dictionary value) min-max-normalized individually.

        :param features_dict:   dictionary of the form <signature_filename> -> <list of feature vectors>
        :return:                same dictionary, with each signature min-max-normalized individually
        """
        normalized_fvd = {}
        for key, signature in features_dict.items():
            max_values, min_values = self._find_feature_max_min(signature)
            minmax = max_values - min_values
            normalized_signature = []
            for feature_vector in signature:
                normalized_feature_vector = (np.array(feature_vector) - min_values) / minmax
                normalized_signature.append(list(normalized_feature_vector))
            normalized_fvd[key] = normalized_signature
        return normalized_fvd

    def _find_feature_max_min(self, signature):
        """
        Finds the maximum values for each of the EXPECTED_NUM_FEATURES features for a single signature.

        :param signature:   a list of feature vectors representing an individual signature (i.e. one entry in a features
                            dictionary
        :return:            maximum values for each of the EXPECTED_NUM_FEATURES features
        """
        dataframe = pd.DataFrame(signature)
        max_values = dataframe.max(axis=0)
        min_values = dataframe.min(axis=0)
        max_values_list = list(max_values)
        min_values_list = list(min_values)
        assert len(max_values_list) == self.EXPECTED_NUM_FEATURES
        return np.array(max_values_list), np.array(min_values_list)
