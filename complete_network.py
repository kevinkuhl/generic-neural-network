import sigmoid
import relu
import parameters
import forward_propagation
import cost
import back_propagation
import numpy as np

# Function to train the model and get the trained parameters
def train_model(X, Y, network_dimensions, learning_rate, training_iterations):
    # Initialize the parameters
    parametersNet = parameters.initialize_parameters(network_dimensions)

    # Iterate over the amount specified, training the network
    for i in range(0, training_iterations):

        '''
        For each iteration, get the network output
        calculate the cost, then compute the gradients
        and update the parameters
        '''

        AL, set_of_caches = forward_propagation.model_forward(X, parametersNet)
        costNet = cost.cross_entropy_cost(AL, Y)
        grads = back_propagation.model_backward(AL, Y, set_of_caches)
        
        parametersNet = parameters.update_parameters(parametersNet, grads, learning_rate)
    
    # At the end, return the parameters
    return parametersNet

# Make testing True when predicting against labelled data
def predict(X, parameters, Y=None, confidence = 0.5, testing=False):
    
    network_output, cache = forward_propagation.model_forward(X, parameters)
    final_predictions = 1 * (network_output > confidence)
    

    if testing:
        labels = Y.reshape(final_predictions.shape)
        accuracy = (np.dot(Y,final_predictions.T) + np.dot(1-Y,1-final_predictions.T))/float(Y.size)
        accuracy = np.squeeze(accuracy)
        print ("Accuracy: {}%".format(accuracy*100))

    return final_predictions