import numpy as np
import time
from sklearn import svm

# SVM - Taha Sukru Karabacakoglu

#Loading csv data to numpy arrays
def loadData(file):
    csv = open(file,"r")
    array = []
    for line in csv:
        array.append(line.strip().split(","))
    array = np.asarray(array, dtype=np.int)
    samples = array[:,1:]
    labels = array[:,0]
    return labels, samples

#Loading Training Set
def loadTrainSet():
    return loadData("train.csv")

#Loading Test Set
def loadTestSet():
    return loadData("test.csv")

#Application
print ("Application started. Loading data...")
start = int(round(time.time() * 1000))
y_train, X_train = loadTrainSet()
print ("Training set loaded with data size: ", len(X_train))
y_test, X_test = loadTestSet()
print ("Test set loaded with data size: ", len(X_test))
print ("Data loaded in ",int(round(time.time() * 1000)) - start," ms. ")

print ("Starting with model generation.")
calcStart = int(round(time.time() * 1000))
clf = svm.SVC(kernel='linear', C = 0.0001)
#clf = svm.SVC(kernel='rbf', gamma=0.001, C = 1.0)
clf.fit(X_train, y_train)
print ("Model generated. Time taken for calculations: ",int(round(time.time() * 1000)) - calcStart, " ms")

print("Accuracy: ", clf.score(X_test,y_test))

print ("Application finished in : ",int(round(time.time() * 1000)) - start, " ms")
