#
# Main file.
#

import os
from parse import Parser
from features import Features
from user import User
import output

# part 1: Read signature properties, calculate feature vectors.
enrollment_signatures = []
verification_signatures = []

pathToProvidedData = "SignatureVerification/"
pathToEnrollmentData = pathToProvidedData + "enrollment/"
pathToVerificationData = pathToProvidedData + "verification/"

print("loading data, generating features")

# enrollment_raw_data is a dictionary. The key is the txt file name of the signature. The value is an array which includes an an array
# for each timestamp with the properties for this timestamp.
enrollment_raw_data = Parser.read(pathToEnrollmentData)
#enrollment_raw_data["001-g-01.txt"]

# enrollment_features is a dictionary. The key is the txt file name of the signature. The value is an array which includes an array
# for each timestamp with the feature for that timestamp.
enrollment_features = Features().generate_features(enrollment_raw_data)
#enrollment_features["001-g-01.txt"])

verification_raw_data = Parser.read(pathToVerificationData)
#verification_raw_data["001-01.txt"]
verification_features = Features().generate_features(verification_raw_data)
#verification_features["001-01.txt"]

# get ground truth with classes g/f
ground_truth = Parser.get_ground_truth(pathToProvidedData + "/gt.txt")
#ground_truth '030-45.txt': 'g'

# get ground truth with classes True/False (True <-> "g", False <-> "f")
ground_truth_boolean = Parser.get_ground_truth(pathToProvidedData + "/gt.txt", use_boolean=True)
#ground_truth_boolean '030-45.txt': 'True'

# normalize enrolment and verification features
print("normalizing features")
enrollment_features_normalized = Features().normalize_signature_features(enrollment_features)
verification_features_normalized = Features().normalize_signature_features(verification_features)

# load all users, let them collect their enrolment signatures (happens in User)
print("loading users")
users = {}
with open(pathToProvidedData + "/users.txt") as users_file:
    for line in users_file:
        users[line.strip()] = User(line.strip(), enrollment_features_normalized)
print("done")

#user can now calculate dissimilarity of a signature w.r.t. to their enrolment signatures
#Getting results TODO slow, look for Mean AP

signatures_file = open("signatures.txt", "w")

for user in users:
    dissimilarities = {}
    verLoc = 0
    for verification in verification_features_normalized:
        dissimilarities[verification] = users[user].calculate_signature_dissimilarity(verification_features_normalized[verification])
        verLoc += 1
        if(verLoc % 50 == 0):
            print("Processed " + str(verLoc) + " verification signatures.")
            break # break for now        
    
    print("Result :")
    signatures_file.write(output.print_dissimilarities(user, dissimilarities) + "\n" )
    break # break for now

signatures_file.close()

