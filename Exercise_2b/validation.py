"""
Team: Over9000

Python Version: 3.6

Pattern Recognition - Exercise 2b
"""

import csv, time
import numpy as np
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------------------------------------------------
# DATA PREPARATION

print()
print('>> start data preparation')

# import train data set
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    train = list(reader)

# import test data set
with open('mnist_test.csv', 'r') as f:
    reader = csv.reader(f)
    test = list(reader)

# convert the lists 'train' and 'test' to (integer) numpy arrays
train = np.array(train, dtype=int)
test = np.array(test, dtype=int)

# extract the features of the training samples
trainFeatures = train[:, 1:]

# extract the labels of the training samples
trainLabels = train[:, 0]

# normalize the labels to be in the range [0,1]
trainFeatures = trainFeatures/255
test = test/255

# We no longer need the samples with labels and the objects for import
del train, reader, f

print('>> done preparing data')


# ---------------------------------------------------------------------------------------------------------------------
# MULTILAYER PERCEPTRON WITH FIXED PARAMETERS

start = time.time()

# set up the classifier
mlp = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='sgd', learning_rate='constant',
                           max_iter=200, tol=0.0001, learning_rate_init=0.1, verbose=False)

# train the classifier
mlp.fit(trainFeatures, trainLabels)

# predict labels for test set
pred_labels = mlp.predict(test)

end = time.time()

print('Time elapsed: ' + repr(end-start) + ' s')

file = open('output.txt', 'w')

for i in range(test.shape[0]):
    file.write(repr(i+1) + ' ' + repr(pred_labels[i]) + '\n')

file.close()
