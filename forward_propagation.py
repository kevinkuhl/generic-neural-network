import numpy as np
import sigmoid
import relu
''' This will define relevant functions related to
the forward propagation
'''

# Defining the linear transformation to be applied to the input of each layer
def linear_transformation(layer_input, W, b):
    Z = np.dot(W,layer_input) + b
    # This will be useful for the backpropagation part
    cache = (layer_input, W, b, Z)

    return Z, cache

# Defining the function which will calculate the activation array of a given layer
def activation(layer_input, W, b, activation_function):

    if activation_function == "relu":
        Z, cache = linear_transformation(layer_input, W, b)
        # A will be the activation array of a given layer (also seen as the input for next layer)
        A = relu.relu(Z)
    
    elif activation_function == "sigmoid":
        Z, cache = linear_transformation(layer_input, W, b)
        A = sigmoid.sigmoid(Z)

    return A, cache

# Lets construct the forward part of the model
def model_forward(X, parameters):
    # List that will contain all the caches produced on the forward step by each layer
    set_of_caches = []

    # Parameters have W and b, therefore, to obtain the total number of layers
    number_of_layers = len(parameters) // 2

    # Let's call the input X is the activation of the layer 0
    A = X
    '''
    ReLu works better overall than Sigmoid, therefore
    we will use sigmoid only on the output layer (to produce a 0 or 1 output)
    '''
    for i in range(1, number_of_layers):
        last_layer_output = A
        A, cache = activation(last_layer_output, parameters['W' + str(i)], parameters['b' + str(i)], "relu")
        set_of_caches.append(cache)
    
    # For the last layer, we use sigmoid
    network_output, cache = activation(A, parameters['W' + str(number_of_layers)], parameters['b' + str(number_of_layers)], "sigmoid" )
    set_of_caches.append(cache)

    return network_output, set_of_caches