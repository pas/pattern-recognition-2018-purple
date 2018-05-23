##
#
# This trains the learning network with the choosen
# optimized parameters with the full train set and
# then classifies the mnist_test set.
#
##

from train import Train
from data_loader import load_train_data, load_test_data
import numpy as numpy

# Chosen optimized parameters
hidden_layer_size = 60
learning_rate = 0.01
num_epochs = 200

train_set_str = "evaluation/train.csv"
test_set_str = "evaluation/mnist_test.csv"

train_x, train_y = load_train_data( train_set_str )
data = ( train_x, train_y )

train = Train( num_epochs , learning_rate , hidden_layer_size , 0, 0 )

train.train( train_x, train_y )

test_x = load_test_data( test_set_str )

classification = train.classify(test_x)

f = open('evaluation/classification.txt', 'w')
i = 1
for c in numpy.nditer(classification):
    f.write("test_ID" + str(i) + ", " + numpy.array_str(c) + "\n")
    i = i+1
f.close()
