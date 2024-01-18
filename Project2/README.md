OVERVIEW OF CODES FOR PROJECT 2


GRADIENT_DESCENT_EXAMPLE.py

    Code snippet that performs gradient descent and stochastic gradient descent on a continuous function for optimizer methods
    GDM (gradient descent with momentum), AdaGrad, RMSprop, ADAM. The code plots a side by side comparison of GD and SGD with
    different optimizers, then performs a grid search for optimal learning rates and penalty factors under ADAM optimization 
    and finally performs a regression on the data set with ADAM under optimal parameters


GRADIENT_DESCENT_LIBRARY.py 

    Holds code for gradient descent and stochastic gradient descent


NeuralNetwork.py

    Object Oriented Code for creating a neural network with a flexible number of layers and nodes in each layer.

    Data file 'data.csv' must be directly available in the same destination as the destination path of the code NeuralNetwork.py

gates_example.py

    Training example using XOR and AND gate.

regression_example.py

    Training example using f(x) = exp(-x**2) + 1.5 * exp(-(x-2)**2) + noise as data generator for regression. 

cancer_data_example.py

    Training example using breast cancer data to perform binary classification (benign or malignant tumor)
