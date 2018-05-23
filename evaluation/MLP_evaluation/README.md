# Evaluation MLP

## Directory

The directory is build in the following way

    |- data # The train data to train the MLP and the test set (mnist) which needs to be classified.
    |- results # The classifaction of the test set in the required format (ASCII plaintext with one line per test sample)

## Training and Evaluation

Via execution of
   $python evaluation.py

The MLP gets trained and then the test data set gets classified. The output is stored in the results folder.
