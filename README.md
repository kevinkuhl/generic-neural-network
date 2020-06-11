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
* Initizalize Parameters: Function that gets as argument the network dimensions on each layer (number of units) and return a dictionary of initialized parameters (weights as random arrays and biases as zeros)
* Update Parameters: Function that gets as argument the current parameters, the gradients for each layer and the learning rate, and returns the new parameters after one iteration of update.


##### cost.py
* Implements the cross entropy cost function considering the total number of training examples used.


##### forward_propagation.py
* Linear Transformation: Function that gets as argument the weight, bias and layer input (X) and returns <img src="https://render.githubusercontent.com/render/math?math=Z=WX%2Bb"> and also a cache containing the layer input, W, b and the linear transformation result (Z), that will be needed in the backpropagation step.
* Activation: Function that gets as argument the layer input, weights, bias and type of activation function and apply the activation function ("relu" or "sigmoid" as a part of function argument).
* Model Foward: Function that builds the model with the parameters from the dimensions specified by the user. If there are N layers, the model will be implemented with N-1 ReLu activated layers and 1 Sigmoid activated layer (as we are dealing with binary classification). ReLu works better than Sigmoid, as it has a constant learning rate of 1 (for x>0), but at the last layer we want an output between 0 and 1, therefore we use Sigmoid.


##### back_propagation.py
* Linear Back Propagation: Function that gets as argument the gradient with respect to the linear transformation (dZ) of the current layer, as well as the activation array from the last layer, weights and biases from the current layer. It returns the gradient with respect to the previous activation array (dprevious_A) and also with respect to the weigths and biases from the current layer.
* Activation Back Propagation: Function that gets as argument the gradient with respect to the current activation array (from current layer), as well as the weights, biases from the current layer and the type of activation function ("relu" or "sigmoid"), calculates dZ and use the above function to return the gradient with respect to the previous activation array (dprevious_A) and also with respect to the weigths and biases from the current layer.
* Model Backward: Function that gets as argument the network output, the true labels and the set of caches (Z, W, b and A) from each layer, calculates the cost and apply the backpropagation functions defined above to calculate the gradients with respect to all needed variables. At the end, it returns the gradients as a dictionary. These gradients will be used as argument to the function that updates the parameters of the network.


##### complete_network.py
* Train Model: Function that gets as argument the network input, the true labels, the network dimensions, learning rate and the number of iterations user wants to train with. After iteration over the forward and backpropagation and updating the parameters, the function will return the correct parameters for the built network with the given set of data.
* Predict: Function that will predict the classification given an input. The input must follow the description given bellow. If the user inserts a list of true labels related to the input, this function will also tell the accuracy of the model for the given dataset. It is also possible to change the threshold value, above which the classification is done (usually 0.5).

## How to use it with images and other multidimensional inputs
A single training example is supposed to look like a column array. It means that if you have an input that has more than one column, you must flatten it first, like shown in the image below.

<img src="https://www.researchgate.net/profile/Budiman_Minasny/publication/334783857/figure/fig4/AS:786596169269249@1564550549811/Illustration-of-flatten-layer-that-is-connecting-the-pooling-layers-to-the-fully.png">

Therefore, if you have M training examples of (128,128,3)(which means RGB images 128 px by 128 px) each, you will end up with a input array of (49152,M). Each of the M columns representing one training example.

For a detailed way of doing that transformation, please reffer to one of the image examples in the 'examples' directory.
