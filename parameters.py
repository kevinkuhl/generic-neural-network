import numpy as np
'''
'dimensions' is a list with the number of units in each layer
(e.g. [4 3 4 1] for 4 input units on the first hidden layer, 4 hidden units on the second hidden layer)
and 1 output unit
''' 
# This will initialize (e.g. W1 and b1 for layer 1) weights as random arrays and bias as zeros
def initialize_parameters(dimensions):
    parametersNet = {}
    number_of_layers = len(dimensions)

    for i in range(1, number_of_layers):
        # Weights will be multiplied by last output activation array (e.g. Wx + b)
        parametersNet["W" + str(i)] = np.random.randn(dimensions[i],dimensions[i-1]) * 0.01
        parametersNet["b" + str(i)] = np.zeros((dimensions[i], 1))
    
    return parametersNet

# Function that will update parameters after the backpropagation
def update_parameters(current_parameters, gradients, learning_rate):
    number_of_layers = len(current_parameters) // 2
    parameters = {}
    for i in range(number_of_layers):
        parameters["W" + str(i+1)] = current_parameters["W" + str(i+1)] - learning_rate*gradients['dW'+str(i+1)]
        parameters["b" + str(i+1)] = current_parameters["b" + str(i+1)] - learning_rate*gradients['db'+str(i+1)]
    
    return parameters