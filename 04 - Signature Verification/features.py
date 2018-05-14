##
#
# Calculates the feature vectors.
#
##

from math import sqrt
import pandas as pd

class Features:

    EXPECTED_NUM_FEATURES = 4

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

    def normalize_feature_vectors(self, feature_vectors_dict):
        """
        Takes as input a dictionary with lists of feature vectors as values, returns the same dictionary with each
        feature vector normalized with respect to its maximum. Note: normalization is done over individual
        feature vectors (i.e. list of 4 features).

        :param feature_vectors_dict:    dictionary of the form <signature_filename> -> <list of feature vectors>
        :return:                        same dictionary, with each feature vector normalized w.r.t. its maximum
        """
        normalized_fvd = {}
        for key, feature_vec_list in feature_vectors_dict.items():
            normalized_list = []
            for feature_vec in feature_vec_list:
                normalized_feature_vec = []
                max_val = max(feature_vec)
                for feature in feature_vec:
                    normalized_feature_vec.append(feature / max_val)
                normalized_list.append(normalized_feature_vec)
            normalized_fvd[key] = normalized_list
        return normalized_fvd

    def normalize_signature_features(self, features_dict):
        """
        Takes as input a dictionary with lists of feature vectors as values, returns the same dictionary with each
        signature (i.e. each dictionary value) normalized individually.

        :param features_dict:   dictionary of the form <signature_filename> -> <list of feature vectors>
        :return:                same dictionary, with each signature normalized individually
        """
        normalized_fvd = {}
        for key, signature in features_dict.items():
            max_values = self._find_feature_maxima(signature)
            print(max_values)


    def _find_feature_maxima(self, signature):
        """
        Finds the maximum values for each of the EXPECTED_NUM_FEATURES features for a single signature.

        :param signature:   a list of feature vectors representing an individual signature (i.e. one entry in a features
                            dictionary
        :return:            maximum values for each of the EXPECTED_NUM_FEATURES features
        """
        dataframe = pd.DataFrame(signature)
        max_values = dataframe.max(axis=0)
        max_values_list = list(max_values)
        assert len(max_values_list) == self.EXPECTED_NUM_FEATURES
        return max_values_list