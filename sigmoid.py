import numpy as np

# Sigmoid definition
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Sigmoid derivative
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

# Backpropagation step for sigmoid
def sigmoid_backpropagation(dA, cache):
    Z = cache[3]
    dZ = dA * sigmoid_derivative(Z)
    return dZ