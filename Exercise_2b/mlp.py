"""
Team: Over9000

Python Version: 3.6

Pattern Recognition - Exercise 2b
"""

import csv, time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------------------------------------------------------
# DATA PREPARATION

print()
print('>> start data preparation')

# import train data set
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    train = list(reader)

# import test data set
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    test = list(reader)

# convert the lists 'train' and 'test' to (integer) numpy arrays
train = np.array(train, dtype=int)
test = np.array(test, dtype=int)

# extract the features of the test and training samples
testFeatures = test[:, 1:]
trainFeatures = train[:, 1:]

# extract the labels of the test and training samples
testLabels = test[:, 0]
trainLabels = train[:, 0]

# normalize the labels to be in the range [0,1]
trainFeatures = trainFeatures/255
testFeatures = testFeatures/255

# We no longer need the samples with labels and the objects for import
del train, test, reader, f

print('>> done preparing data')

# ---------------------------------------------------------------------------------------------------------------------
# IF YOU WANT TO USE ONLY PART OF THE DATA

trainFeatures = trainFeatures[:1000, :]
trainLabels = trainLabels[:1000]
testFeatures = testFeatures[:100, :]
testLabels = testLabels[:100]

# ---------------------------------------------------------------------------------------------------------------------
# MULTILAYER PERCEPTRON

start = time.time()

# number of sets in cross validation:
cv_num = 5

# parameters to test in grid search:
params = {"hidden_layer_sizes": np.linspace(10, 100, 10, dtype=int), 'learning_rate_init': np.linspace(0.1, 1, 10)}
# TODO: Optimize number of training iterations (during learning; backpropagation).
# TODO: Plot a graph showing the error on the training set and the validation set, respectively, with respect to the training epochs.
# note: error = 1 - accuracy

# the classifier
mlp = MLPClassifier(solver='sgd', learning_rate='constant')

# grid search
gs_mlp = GridSearchCV(mlp, param_grid=params, scoring='accuracy', n_jobs=2, cv=cv_num)
gs_mlp.fit(trainFeatures, trainLabels)
# note: in the end, the mlp is refit with the best estimator and the entire data set

# read tested parameter combinations and their scores
params_gs = gs_mlp.cv_results_['params']
score_cv = gs_mlp.cv_results_['mean_test_score']

# output tested parameters and scores
print()
print(' nn | lr  | acc')
print('------------------')
for i in range(len(params_gs)):
    print('{:3.0f} | {:1.1f} | {:1.4f}'.format(params_gs[i]['hidden_layer_sizes'], params_gs[i]['learning_rate_init'], score_cv[i]))
print()

# predict labels for test set
# TODO: Perform the random initialization several times and choose the best network during cross-validation.
pred_labels = gs_mlp.predict(testFeatures)
accuracy_cv = metrics.accuracy_score(pred_labels, testLabels)

end = time.time()

best_num_neur = gs_mlp.best_params_['hidden_layer_sizes']
best_learn_rate =gs_mlp.best_params_['learning_rate_init']

print("Accuracy of MLP with optimal number of neurons = {:1.0f} and optimal learning rate = {:1.1f} on training set is {:1.4f}".format(best_num_neur, best_learn_rate, accuracy_cv))
print('Time elapsed: ' + repr(end-start) + ' s')
print()



