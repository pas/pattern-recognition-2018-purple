# Multilayer Perceptron Draft Tensorflow

Author: Silas<br>
Deep Learning Library: Tensorflow (Google)

## How to Use
- create new virtual environment, install dependencies (requirements.txt to follow)
- create a new empty folder MNIST_data (will change soon, will need to provide data from assignment)
- run multilayer_perceptron.py from virtualenv, will train and evaluate the model with a fixed set of hyperparameters

## References
This prototype is based on the following resources:

- [Tensorflow: MNIST for ML Beginners](https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners)
- [Siraj Raval: Tensorflow in 5 Minutes (YouTube)](https://www.youtube.com/watch?v=2FmcHiLCwTU)
- [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/)

## About Tensorflow
Tensorflow is a library for numerical computation using data flow graphs. It is built to scale and can run operations on both the CPU and the GPU. Tensorflow is currently one of the most widely used deep learning libraries in research and industry.

In Tensorflow, every operation is represented as a node in a graph. Such operations can be simple matrix multiplications, argmax functions, sigmoid or softmax functions, etc., or more complex functions such as a Gradient Descent optimizer that optimizes the value of a node by adjusting all variables in the graph. Special operations are `placeholder` and `Variable`. Both are n-dimensional tensors, which means they can be scalars, vectors, matrices, or multi-dimensional matrices. When creating a placeholder or variable, we therefore need to define its shape. A variable is initialized with default values and can be changed during graph execution. A placeholder needs to be filled when the graph is evaluated, and its value doesn't change during graph evaluation.

Graphs are lazy-evaluated. That is, we first construct the graph, at which point, no computations take place. We then create a so-called session, in which we fill the necessary placeholders with explicit values and evaluate a specific node of the graph. Tensorflow will then evaluate this node and all nodes that feed into it, and return a numpy array.

Behind the scenes, Tensorflow always has a default graph. That means, if we create a new node (variable, placeholder, other operation, ...) in our code, this node is added to the default graph. Not all nodes in a graph need to be connected. We can also work on multiple graphs and set a different graph as default, which means newly created operations will be added to that graph instead.

In summary, Tensorflow is primarily a library for numerical computations. However, it also includes many functions that are handy for machine learning purposes, such as the sigmoid function and the Gradient Descent optimizer.
