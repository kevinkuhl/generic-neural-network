'''
Before executing, be sure to put all the files in the same directory
(files from network and example)
'''

import complete_network
import numpy as np
from sklearn.model_selection import train_test_split

f = open("examples/haberman_survival/haberman.data")

dataset = []
labels = []

for line in f.readlines():
    value = line.split(",")
    array = np.array([[int(value[0])],
       [int(value[1])],
       [int(value[2])]])
    label = np.array([[int(value[3]) - 1]])
    labels.append(label)
    dataset.append(array)

dataset = np.concatenate(dataset, axis=1)
labels = np.concatenate(labels, axis=0)

X_train, X_test, y_train, y_test = train_test_split(dataset.T, labels, test_size=0.33)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

# Feel free to modify the dimensions, keeping the first and last one fixed
dimensions = [X_train.shape[0], 10, 15, 10, 1]

parameters = complete_network.train_model(X_train, y_train,dimensions,0.009, 5000)

complete_network.predict(X_test, parameters, y_test,testing=True)