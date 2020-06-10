import numpy as np

# ReLu definition
def relu(x):
    return np.maximum(0,x)

# ReLu derivative
def relu_derivative(x):
    return 1 * (x > 0)

# Backpropagation step for ReLu
def sigmoid_backpropagation(dA, cache):
    Z = cache[3]
    dZ = dA * relu_derivative(Z)
    return dZ