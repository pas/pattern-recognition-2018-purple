#
# Main file.
#

import os
from parse import Parser
from features import Features

# part 1: Read signature properties, calculate feature vectors.
enrollment_signatures = []
verification_signatures = []

pathToProvidedData = "SignatureVerification/"
pathToEnrollmentData = pathToProvidedData + "enrollment/"
pathToVerificationData = pathToProvidedData + "verification/"

# enrollment_raw_data is a dictionary. The key is the txt file name of the signature. The value is an array which includes an an array
# for each timestamp with the properties for this timestamp.
enrollment_raw_data = Parser.read(pathToEnrollmentData)

# enrollment_features is a dictionary. The key is the txt file name of the signature. The value is an array which includes an array
# for each timestamp with the feature for that timestamp.
enrollment_features = Features().generate_features(enrollment_raw_data)

verification_raw_data = Parser.read(pathToVerificationData)
verification_features = Features().generate_features(verification_raw_data)

# get ground truth with classes g/f
ground_truth = Parser.get_ground_truth(pathToProvidedData + "/gt.txt")

# get ground truth with classes True/False (True <-> "g", False <-> "f")
ground_truth_boolean = Parser.get_ground_truth(pathToProvidedData + "/gt.txt", use_boolean=True)

enrollment_features_normalized = Features().normalize_signature_features(enrollment_features)
verification_features_normalized = Features().normalize_signature_features(verification_features)

print(enrollment_features)
