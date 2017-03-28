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
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings

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

# trainFeatures = trainFeatures[:100, :]
# trainLabels = trainLabels[:100]
# testFeatures = testFeatures[:10, :]
# testLabels = testLabels[:10]


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# SIMPLE MULTILAYER PERCEPTRON WITH FIXED PARAMETERS (as illustrative example)

# start_simple = time.time()
#
# # set up the classifier
# mlp_simple = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='sgd', learning_rate='constant',
#                            max_iter=200, tol=0.0001, learning_rate_init=0.1, verbose=False)
# # train the classifier
# with ignore_warnings(category=ConvergenceWarning):
#     mlp_simple.fit(trainFeatures, trainLabels)
# # predict labels for test set
# pred_labels_simple = mlp_simple.predict(testFeatures)
# # compute accuracy
# accuracy_simple = metrics.accuracy_score(pred_labels_simple, testLabels)
#
# end_simple = time.time()
#
# print("Accuracy of MLP on training is {:1.4f}".format(accuracy_simple))
# print('Time elapsed: ' + repr(end_simple-start_simple) + ' s')
#
# # plot the loss function during learning
# plt.plot(mlp_simple.loss_curve_)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# MULTILAYER PERCEPTRON WITH CROSS-VALIDATION (the actual exercise)

# SETTINGS:
# ---------

# number of sets in cross validation:
cv_num = 5  # this will be used for all cross validations performed

# parameters to test in first grid search:
params1 = {"hidden_layer_sizes": np.linspace(10, 100, 10, dtype=int), 'learning_rate_init': np.linspace(0.1, 1, 10)}

# number of times the random initialization should be done:
num_rand_init = 10

# parameters to test in second grid search:
params2 = {'random_state': np.random.choice(100, size=num_rand_init, replace=False)}


# FIRST GRID SEARCH (FOR NUMBER OF NEURONS AND LEARNING RATE):
# ----------------––------------------------------------------

# the classifier
mlp1 = MLPClassifier(activation='relu', solver='sgd', learning_rate='constant', max_iter=200, random_state=42,
                     tol=0.0001, verbose=False)

start1 = time.time()

# do grid search
gs_mlp1 = GridSearchCV(mlp1, param_grid=params1, scoring='accuracy', n_jobs=-1, cv=cv_num, refit=False)
gs_mlp1.fit(trainFeatures, trainLabels)
# note: in the end, the mlp is not refit with the best estimator and the entire data set (saves time)
# we have to optimize the number of iterations first and additionally do the random initialization several times.

end1 = time.time()

# read tested parameter combinations and their scores
params_gs1 = gs_mlp1.cv_results_['params']
score_gs1 = gs_mlp1.cv_results_['mean_test_score']

# output tested parameters and scores
print()
print(' nn | lr  | acc')
print('------------------')
for i in range(len(params_gs1)):
    print('{:3.0f} | {:1.1f} | {:1.4f}'.format(params_gs1[i]['hidden_layer_sizes'], params_gs1[i]['learning_rate_init'],
                                               score_gs1[i]))
print()
print('Time elapsed: ' + repr(end1-start1) + ' s')
print()

# read best parameters
best_num_neur = gs_mlp1.best_params_['hidden_layer_sizes']
best_learn_rate = gs_mlp1.best_params_['learning_rate_init']


# OPTIMIZE NUMBER OF TRAINING ITERATIONS AND PLOTTING:
# ----------------––----------------------------------

# TODO: Optimize number of training iterations (max_iter) and plot a graph showing the error on the training set and the validation set, respectively, with respect to the training epochs.
best_num_it = 200  # place holder, this should be calculated here.
# note: for this part, the convergence criteria (tol) should be set very low (e.g. 0.0000001) so that the learning stops
# due to max_iter and not due to tol. also include the "ignore warnings" part from the simple example above, or else
# a warning will be shown any time the learning stops due to max_iter.
# for plotting: see last line in the simple example above.

# Suggestion: since controlling the convergence with the tolerance works better as with number of iteration,
# we should just plot the loss function several times and check after how many iterations we reach convergence...


# SECOND GRID SEARCH (FOR NEURON WEIGHTS DUE TO DIFFERENT RANDOM INITIALIZATIONS):
# ----------------––--------------------------------------------------------------

# the classifier
mlp2 = MLPClassifier(hidden_layer_sizes=best_num_neur, activation='relu', solver='sgd', learning_rate='constant',
                     max_iter=best_num_it, tol=0.0001, learning_rate_init=best_learn_rate, verbose=False)

start2 = time.time()

# grid search
gs_mlp2 = GridSearchCV(mlp2, param_grid=params2, scoring='accuracy', n_jobs=-1, cv=cv_num, refit=True)
gs_mlp2.fit(trainFeatures, trainLabels)
# note: in the end, the mlp is refit with the best estimator and the entire data set, so that prediction of the
# test set is directly possible

end2 = time.time()

# read tested parameter combinations and their scores
params_gs2 = gs_mlp2.cv_results_['params']
score_gs2 = gs_mlp2.cv_results_['mean_test_score']

# output tested parameters and scores
print(' rs | acc')
print('------------------')
for i in range(len(params_gs2)):
    print('{:3.0f} | {:1.4f}'.format(params_gs2[i]['random_state'], score_gs2[i]))
print()
print('Time elapsed: ' + repr(end2-start2) + ' s')
print()


# PREDICT LABELS FOR TEST SET AND COMPUTE ACCURACY:
# -------------------------------------------------

start3 = time.time()

pred_labels = gs_mlp2.predict(testFeatures)
tot_accuracy = metrics.accuracy_score(pred_labels, testLabels)

end3 = time.time()

print("Accuracy of MLP with optimal number of neurons = {:1.0f} and optimal learning rate = {:1.1f} on training "
      "set is {:1.4f}".format(best_num_neur, best_learn_rate, tot_accuracy))
print('Time elapsed: ' + repr(end3-start3) + ' s')
print()



