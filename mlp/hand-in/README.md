# Project 2b

## Directory

The directory is build in the following way

    |- data # All tested or validated data
    |- folds # The constructed folds (was build with split_data.py)
    |- plots # All plots for the different hyper parameters
     |- full # Plots with the means of all folds
     |- folds # Plots with data of each fold. Each full training is put on one plot
    |- results # Temporary cache for results. Used to create the full plots
    |- helpers # Holds python files to help in the process (like creating partitions for the cross-valdidation)

## Validation
For each set of validated hyper parameters we used a 4-fold-cross-validation and used
values learning rates of 0.1, 0.05, 0.011, 0.01 and 0.009 and size of layer of
20, 40, 60 and 80.

You can run a full cycle with all the learning rates and the size of the 
hidden layers stored in runner.py use (-e for number of epochs and
-f for number of folds):

    runner.py -e 100 -f 4
    
This results in plots in plots/full and plotts/folds and stores the best result for each learning rate
and the corresponding size of the hidden layer into the results.txt file.

## Results

Out of the results.txt you can read the best setup for the parameters by choosing the one with
the highest accuracy and lowest standard deviation. In our case this was:

0.01,60,100,0.9006631549999999,0.008368006144851625

At the end you can check we checked our solution with final.py where we used a learning rate
of 0.01 with a size of 60 for the layer. The parameters were directly written into the final.py
file.

    final.py

If we look at the plot for learning rate 0.1 and 60 hidden layer, we can observe that
the accuracy of both, the test and the validation set increases rapidly with the first training epochs.
Then the accuracy of the test set is always higher than that of the validation set.

learning rate: 0.01
hidden layers: 60
number of epochs: 200
accuracy: 0.9163673 
