#
# Main file.
#
import os
import time
from parse import Parser
from features import Features
from user import User
import output
from multiprocessing import Pool

total_time = time.time()

# part 1: Read signature properties, calculate feature vectors.
enrollment_signatures = []
verification_signatures = []

# This path was used during the development.
#pathToProvidedData = "SignatureVerification/"
# This path is used for the evaluation.
pathToProvidedData = "Evaluation/TestSignatures/"
pathToEnrollmentData = pathToProvidedData + "enrollment/"
pathToVerificationData = pathToProvidedData + "verification/"

print("loading data, generating features")

# enrollment_raw_data is a dictionary. The key is the txt file name of the signature.
# The value is an array which includes an array
# for each timestamp with the properties for this timestamp.
enrollment_raw_data = Parser.read(pathToEnrollmentData)
#enrollment_raw_data["001-g-01.txt"]

# enrollment_features is a dictionary. The key is the txt file name of the signature.
# The value is an array which includes an array
# for each timestamp with the feature for that timestamp.
enrollment_features = Features().generate_features(enrollment_raw_data)
#enrollment_features["001-g-01.txt"])

verification_raw_data = Parser.read(pathToVerificationData)
#verification_raw_data["001-01.txt"]
verification_features = Features().generate_features(verification_raw_data)
#verification_features["001-01.txt"]

# get ground truth with classes g/f
#ground_truth = Parser.get_ground_truth(pathToProvidedData + "/gt.txt")
#ground_truth '030-45.txt': 'g'

# get ground truth with classes True/False (True <-> "g", False <-> "f")
#ground_truth_boolean = Parser.get_ground_truth(pathToProvidedData + "/gt.txt", use_boolean=True)
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
#Getting results MultiThreaded (5 ver signatures executes in 11 mins for 30 users)
#TODO look for Mean AP

print("Processing...")

def calculateUserDistances(users_index):
    start_time = time.time()
    print("User " + users_index + " executing in thread")
    dissimilarities = {}
    verLoc = 0
    for verification in verification_features_normalized:
        if(users_index not in verification):
            continue
        dissimilarities[verification] = users[users_index].calculate_signature_dissimilarity(verification_features_normalized[verification])
        verLoc += 1
        if(verLoc % 100 == 0):
            print("User " + users_index + " Thread Processed " + str(verLoc) + " verification signatures.")
            #break

    print("--- %s seconds ---" % (time.time() - start_time))
    return output.print_dissimilarities(users_index, dissimilarities) + "\n" 
    
# This path was used during the development.
#signature_directory = "signatures"
# This path is used for the evaluation.
signature_directory = "evaluationSignatures"

if not os.path.exists(signature_directory):
    os.makedirs(signature_directory)

with Pool(processes=4) as pool:
    results = pool.map(calculateUserDistances, users)
    for resultTxt in results:
        signatures_file = open(signature_directory + "/signatures-" + resultTxt[0:3] + ".txt", "w")
        signatures_file.write(resultTxt + "\n")
        signatures_file.close()
    
""" true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for file in os.listdir(signature_directory):
    signatures_file = open(signature_directory + "/" + file, "r")
    for line in signatures_file:
        line_split = line.split(",")
        if(len(line_split) < 2):
            continue
        grount_truth_string = ground_truth.get(line_split[0])
        if(grount_truth_string is not None and grount_truth_string in line_split[2]):
            if(grount_truth_string == "g"):
                true_positive = true_positive + 1
            else:
                true_negative = true_negative + 1
        elif(grount_truth_string is not None):
            if(grount_truth_string == "g"):
                false_negative = false_negative + 1
            else:
                false_positive = false_positive + 1
total = true_positive + true_negative + false_positive + false_negative

print("correct= " + str(true_positive + true_negative))
print("incorrect = " + str(false_positive + false_negative))
print("correct/incorrect = " + str((true_positive + true_negative) / total))
print("\n")

print("true_positive: " + str(true_positive))
print("true_negatives: " + str(true_negative))
print("false_positive: " + str(false_positive))
print("false_negatives: " + str(false_negative))
print("\n")

print("precision = " + str(true_positive / (true_positive + false_positive)))
print("recall = " + str(true_positive / (true_positive + false_negative)))
print("\n")
 """
print("TOTAL TIME: %s seconds ---" % (time.time() - total_time))

# Generating the expected output file for the evaluation.
expected_output = open("Evaluation/results_signature.txt", "w")
for file in sorted(os.listdir(signature_directory)):
    user_result = ""
    signatures_file = open(signature_directory + "/" + file, "r")
    for line in signatures_file:
        line_split = line.split(",")
        if(len(line_split) < 2):
            if "===" in line_split[0]:
                continue
            elif ("\n" == line_split[0]):
                # Arrived at the end of file
                break
            else:
                user_result = "user"+str(line_split[0][:-1])+", "
        else:
            signature_result = "signature_"+line_split[0].split("-")[1]
            dissimilarity_result = line_split[1]
            user_result = user_result+str(signature_result) + "," + str(dissimilarity_result) + ", "
    # Remove the last comma and space
    expected_output.write(str(user_result[:-2]) + "\n")
    #expected_output.seek(-2, os.SEEK_CUR)
    #expected_output.truncate()
    #expected_output.write("\n")
expected_output.close()