import complete_network
import forward_propagation
from PIL import Image
import numpy as np
import os

# directory_train_dogs = "examples/dog_non_dog/dogs"
# directory_train_non_dogs = "examples/dog_non_dog/nondogs"

# trainImages = []
# labels = []

# for filename in os.listdir(directory_train_dogs):
#     image = Image.open(os.path.join(directory_train_dogs, filename)).convert('RGB')
#     image = image.resize((64,64))
#     image.show()
#     image = np.asarray(image)
#     trainImages.append(image)
#     labels.append(1)

# for filename in os.listdir(directory_train_non_dogs):
#     image = Image.open(os.path.join(directory_train_dogs, filename)).convert('RGB')
#     image = image.resize((64,64))
#     image.show()
#     image = np.asarray(image)
#     trainImages.append(image)
#     labels.append(0)

# trainImages = np.asarray(trainImages)
# trainImages = trainImages.reshape(trainImages.shape[0], -1).T
# trainImages = trainImages / 255
# labels = np.asarray(labels)
# labels = labels.reshape(1, labels.shape[0])

# dimensions = [trainImages.shape[0], 10, 15, 1]

# parameters = complete_network.train_model(trainImages, labels, dimensions, 0.001, 10000)

# testImage = Image.open("examples/dog_non_dog/test.jpg")
# testImage = testImage.resize((64,64))
# testImage = np.asarray(testImage)
# testImage = testImage.reshape(12288, 1)
# #prediction = complete_network.predict(testImage, parameters)

# network_output, cache = forward_propagation.model_forward(trainImages, parameters)

# # print(parameters)

# print(network_output)