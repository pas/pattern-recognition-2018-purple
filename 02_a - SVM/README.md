# Project 2a

## How to run

You can open source file to modify parameters cross_validation and svm kernels. If cross validation is enabled, 
then it will perform cross validation on training data. Otherwise, it will run on test set.

    svm.py 
    
Program will output Accuracy or Av. Accuracy of cross validation and execution times.

## Results

We mainly focused on linear and polynomial kernels. We found out that the best performing kernel is polynomial.

Best performing kernel is polynomial with the following values

    gamma: 1.0
    C: 1.0
    degree: 2
	
	Accuracy: 0.9747

Best performing linear kernel is with the following values

    C: 0.000001
	
	Accuracy: 0.9429
	
## Conclusions

As C value increases the execution time of the program increases as well for the linear kernel.
Highest accuracy we can get with sigmoid is around 0.9102 and rbf is around 0.9649.
In comparision of execution times, sigmoid and rbf spends similar time. Polynomial kernel
performs nearly same with linear kernel and they are faster than sigmoid and rbf. 
Best svm kernel for this data set is Polynomial kernel.
