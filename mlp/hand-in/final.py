##
#
# This trains the learning network with the choosen 
# optimized parameters with the full train set and
# against the yet unseen test set.
#
##

from train import Train
from loader import Loader

# Chosen optimized parameters

hidden_layer_size = 70
learning_rate = 0.009
num_epochs = 250

train_set_str = "data/train-orig.csv"
test_set_str = "data/test.csv"

loader = Loader()
(train_x, train_y, test_x, test_y) = loader.from_files(train_set_str, test_set_str)

train = Train( num_epochs , learning_rate , hidden_layer_size , 0 , 0 )
train.start(train_x, train_y, test_x, test_y)
print( train.get_accuracy( test_x , test_y ) )

