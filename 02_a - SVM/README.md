# Project 2a

### VALIDATION - LAST TASK

The code has been slightly altered in order to accomodate the new file format without the labels. The code was extended as to be able to run both formats.

Inside the validation folder the mnist_validation dataset can be found. The program will print all preditctions to standard output and additionally to a file also found in the folder "validation" in following format:

{{ row_id }}: {{ predicted_class }}

row_id [0..9999]
number [0..9]

Short summary of occurences of numbers:
{0: 988, 1: 1145, 2: 1038, 3: 1008, 4: 987, 5: 886, 6: 954, 7: 1026, 8: 972, 9: 996}

## How to run

You can open source file to modify parameters cross_validation and svm kernels. If cross validation is enabled, 
then it will perform cross validation on training data. Otherwise, it will run on test set.

    svm.py 
    
Program will output Accuracy or Av. Accuracy of cross validation and execution times.

## Results

You can look into files for our results

    svm_linear_kernel_results.txt
    svm_polynomial_kernel_results.txt
    svm_other_kernel_results.txt

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

As C value increases the execution time of the program increases as well.

Highest accuracy we can get with sigmoid is around 0.9102 and rbf is around 0.9649.

In comparison of execution times, sigmoid and rbf spends similar time. Polynomial kernel
performs nearly same with linear kernel and they are faster than sigmoid and rbf. 
