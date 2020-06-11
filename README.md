# Generic Neural Network

This project intends to create basic individual building blocks to allow the construction of a neural network for binary classification.

Bellow you will find the description for the created files

##### sigmoid.py
* Sigmoid function
* Sigmoid derivative
* Sigmoid Backpropagation: Function that returns the gradient of the cost with respect to the linear argument of the activation function (dZ). This is based on the use of a cross entropy loss function <img src="https://render.githubusercontent.com/render/math?math=L(y,\hat{y})=-(ylog(\hat{y})%2B(1-y)log(1-\hat{y})">


##### relu.py
* ReLu function
* ReLu derivative
* ReLu Backpropagation: Function that returns the gradient of the cost with respect to the linear argument of the activation function (dZ). This is based on the use of a cross entropy loss function <img src="https://render.githubusercontent.com/render/math?math=L(y,\hat{y})=-(ylog(\hat{y})%2B(1-y)log(1-\hat{y})">


##### parameters.py
* Initizalize Parameters: Function that gets as argument the network dimensions on each layer (number of units) and return a dictionary of initialized parameters (weights as random arrays and bias as zeros)
* Update Parameters: Function that gets as argument the current parameters, the gradients for each layer and the learning rate, and returns the new parameters after one iteration of update.


##### cost.py
* Implements the cross entropy cost function considering the total number of training examples used.


##### forward_propagation.py
* Linear Transformation: Function that gets as argument the weight, bias and layer input (X) and returns <img src="https://render.githubusercontent.com/render/math?math=Z=WX%2Bb"> and also a cache containing the layer input, W, b and the linear transformation result (Z), that will be needed in the backpropagation step.
* Activation: Function that gets as argument the layer input, weights, bias and type of activation function and apply the activation function ("relu" or "sigmoid" as a part of function argument).
* Model Foward: Function that builds the model with the parameters from the dimensions specified by the user. If there are N layers, the model will be implemented with N-1 ReLu activated layers and 1 Sigmoid activated layer (as we are dealing with binary classification). ReLu works better than Sigmoid, as it has a constant learning rate of 1 (for x>0), but at the last layer we want an output between 0 and 1, therefore we use Sigmoid.
