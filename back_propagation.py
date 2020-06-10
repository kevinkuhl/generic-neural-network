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