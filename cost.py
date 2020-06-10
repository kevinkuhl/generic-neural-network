import numpy as np
'''
Here will be defined the function to compute the cost
after a given forward iteration.
Cross entropy will be used, as it is a convex function
for the gradient descent method
'''
# Y contains the true labels
def cross_entropy_cost(network_output, Y):
    number_of_training_ex = Y.shape[1]

    cost = -(1/number_of_training_ex)*np.sum(Y*np.log(network_output) + (1-Y)*np.log(1-network_output), axis=1, keepdims=True)
    cost = np.squeeze(cost)

    return cost