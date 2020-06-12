import sigmoid
import relu
import numpy as np
'''
All backpropagation functions will be defined
taking into account the use of the cross entropy
cost function.
If you choose to use another cost function, be aware
of the need of changing the definition of each gradient.
'''

# Given the gradient of the cost with respect to the linear output of a layer
# Return the gradient of the cost with respect to the activation array of the previous layer
# As well as with respect to the weigths and bias of the current layer
def linear_back_prop(dZ, cache):
    # Calling the last layer output previous_A (previous layer activation array)
    previous_A, W, b = cache[0], cache[1], cache[2]
    number_of_training_ex = previous_A.shape[1]

    dW = (1/number_of_training_ex)*np.dot(dZ,previous_A.T)
    db = (1/number_of_training_ex)*np.sum(dZ, axis=1, keepdims=True)
    dprevious_A = np.dot(W.T, dZ)

    return dprevious_A, dW, db

'''
Given dA and dZ for the current layer,
compute dA of the previous layer (dprevious_A)
and dW, db for the current layer based on the 
activation function used

dA is the gradient of the cost function with
respect to the activation array.
'''
def activation_back_prop(dA, cache, activation_function):
    
    if activation_function == "relu":
        dZ = relu.relu_backpropagation(dA, cache)
        dprevious_A, dW, db = linear_back_prop(dZ, cache)
    
    elif activation_function == "sigmoid":
        dZ = sigmoid.sigmoid_backpropagation(dA, cache)
        dprevious_A, dW, db = linear_back_prop(dZ, cache)

    return dprevious_A, dW, db

# The network output is the same thing as the activation array of the last layer
# Lets call it AL
def model_backward(AL, Y, set_of_caches):
    grads = {}
    number_of_layers = len(set_of_caches)
    number_of_training_ex = AL.shape[1]

    # Assuring that Y has the same shape as network_output
    Y = Y.reshape(AL.shape)

    # Considering that the cost function is the cross entropy
    # Lets consider the activation array of the last layer as AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = set_of_caches[-1]
    grads["dA" + str(number_of_layers-1)], grads["dW" + str(number_of_layers)], grads["db" + str(number_of_layers)] = activation_back_prop(dAL, current_cache, 'sigmoid')

    # Now for the rest of the layers (activated by ReLu functions)
    for l in reversed(range(number_of_layers-1)):
        current_cache = set_of_caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_back_prop(grads["dA"+str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads