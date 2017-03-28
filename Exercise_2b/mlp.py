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
# mlp_simple.fit(trainFeatures, trainLabels)
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
params1 = {'hidden_layer_sizes': np.linspace(10, 100, 10, dtype=int), 'learning_rate_init': np.linspace(0.1, 1, 10)}

# number of times the random initialization should be done:
num_rand_init = 20

# parameters to test in second grid search:
params2 = {'random_state': np.random.choice(100, size=num_rand_init, replace=False)}

# investigated convergence tolerance:
tols = [0.1, 0.01, 0.001, 0.0001, 0.00001]


# FIRST CROSS-VALIDATION (FOR NUMBER OF NEURONS AND LEARNING RATE):
# ----------------––-----------------------------------------------

print()
print('FIRST CROSS-VALIDATION (FOR NUMBER OF NEURONS AND LEARNING RATE)')
print('----------------------------------------------------------------')

# the classifier. we fix a random state for better comparison of the parameters.
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

# read best parameters
best_num_neur = gs_mlp1.best_params_['hidden_layer_sizes']
best_learn_rate = gs_mlp1.best_params_['learning_rate_init']

# output tested parameters and scores
print()
print(' nn | lr  | acc')
print('------------------')
for i in range(len(params_gs1)):
    print('{:3.0f} | {:.1f} | {:.4f}'.format(params_gs1[i]['hidden_layer_sizes'], params_gs1[i]['learning_rate_init'],
                                               score_gs1[i]))
print()
print('nn = number of neurons')
print('lr = learning rate')
print('acc = accuracy')
print()
print('Best parameter combination:')
print('number of hidden layers = ' + repr(best_num_neur))
print('learning rate = {:.1f}'.format(best_learn_rate))
print()
print('Time elapsed: {:.3f} s'.format(end1-start1))


# SECOND CROSS-VALIDATION (FOR TOLERANCE):
# ----------------––----------------------

print()
print('SECOND CROSS-VALIDATION (FOR TOLERANCE)')
print('---------------------------------------')
print()

# to access the attribute loss_curve, we need to perform the cross validation by hand:

# part the indices of the training set into predefined number of sets:
idx = np.array(np.array_split(np.random.permutation(np.arange(len(trainLabels), dtype=int)), cv_num))

fig, axes = plt.subplots(1, cv_num)

start2 = time.time()

acc_tol = list()

for t in range(len(tols)):

    acc = list()
    num_iter = list()
    loss = list()

    axes[t].set_title('tol = ' + repr(tols[t]))
    axes[t].set_xlabel('number of iterations')
    axes[t].set_ylabel('loss')

    for i in range(cv_num):
        x_train = trainFeatures[np.concatenate((np.delete(idx, i, 0))), :]
        y_train = trainLabels[np.concatenate((np.delete(idx, i, 0)))]
        x_test = trainFeatures[idx[i], :]
        y_test = trainLabels[idx[i]]
        mlp2 = MLPClassifier(hidden_layer_sizes=best_num_neur, activation='relu', solver='sgd',
                             learning_rate='constant', max_iter=200, random_state=42,
                             tol=tols[t], learning_rate_init=best_learn_rate, verbose=False)
        mlp2.fit(x_train, y_train)
        y_pred = mlp2.predict(x_test)
        acc.append(metrics.accuracy_score(y_pred, y_test))
        num_iter.append(mlp2.n_iter_)
        loss.append(mlp2.loss_)
        axes[t].plot(mlp2.loss_curve_, label='cv | n_iter = ' + repr(mlp2.n_iter_))

    # do training with entire training set
    mlp2.fit(trainFeatures, trainLabels)
    # predict entire test set
    pred_labels2 = mlp2.predict(testFeatures)
    acc_train = metrics.accuracy_score(pred_labels2, testLabels)
    acc.append(acc_train)
    acc_tol.append(np.mean(acc))
    axes[t].plot(mlp2.loss_curve_, label='train | n_iter = ' + repr(mlp2.n_iter_))
    axes[t].legend()

    # output tested parameters and scores
    print('tolerance = ' + repr(tols[t]))
    print('------------------')
    print()
    print(' ni | loss   | acc')
    print('---------------------')
    for i in range(cv_num):
        print('{:3.0f} | {:.4f} | {:.4f}'.format(num_iter[i], loss[i], acc[i]))

    print()
    print('Training on entire training set needed {:d} iterations, ended with loss {:.4f} '
          'and achieved accuracy {:.4f}.'.format(mlp2.n_iter_, mlp2.loss_, acc_train))
    print()
    print('The mean accuracy is {:.4f}'.format(acc_tol[t]))
    print()

end2 = time.time()

best_tol = tols[np.argmax(acc_tol)]

print('ni = number of iterations')
print('acc = accuracy')
print()
print('Best performing tolerance is ' + repr(best_tol) + ' with accuracy {:.4f}.'.format(max(acc_tol)))
print()
print('Time elapsed: {:.3f} s'.format(end2-start2))


# THIRD CROSS-VALIDATION (FOR NEURON WEIGHTS DUE TO DIFFERENT RANDOM INITIALIZATIONS):
# ----------------––------------------------------------------------------------------

print()
print('THIRD CROSS-VALIDATION (FOR NEURON WEIGHTS DUE TO DIFFERENT RANDOM INITIALIZATIONS)')
print('-----------------------------------------------------------------------------------')
print()

# the classifier
mlp3 = MLPClassifier(hidden_layer_sizes=best_num_neur, activation='relu', solver='sgd', learning_rate='constant',
                     max_iter=200, tol=best_tol, learning_rate_init=best_learn_rate, verbose=False)

start3 = time.time()

# grid search
gs_mlp3 = GridSearchCV(mlp3, param_grid=params2, scoring='accuracy', n_jobs=-1, cv=cv_num, refit=True)
gs_mlp3.fit(trainFeatures, trainLabels)
# note: in the end, the mlp is refit with the best estimator and the entire data set, so that prediction of the
# test set is directly possible

end3 = time.time()

# read tested parameter combinations and their scores
params_gs3 = gs_mlp3.cv_results_['params']
score_gs3 = gs_mlp3.cv_results_['mean_test_score']

# output tested parameters and scores
print(' rs | acc')
print('------------')
for i in range(len(params_gs3)):
    print('{:3.0f} | {:.4f}'.format(params_gs3[i]['random_state'], score_gs3[i]))
print()
print('rs = random state')
print('acc = accuracy')
print()
print('Time elapsed: {:.3f} s'.format(end3-start3))


# PREDICT LABELS FOR TEST SET AND COMPUTE ACCURACY:
# -------------------------------------------------

print()
print('PREDICT LABELS FOR TEST SET AND COMPUTE ACCURACY')
print('------------------------------------------------')
print()

start4 = time.time()

pred_labels = gs_mlp3.predict(testFeatures)
tot_accuracy = metrics.accuracy_score(pred_labels, testLabels)

end4 = time.time()

print('Accuracy of MLP on the entire training set with')
print()
print('  - optimal number of neurons = ' + repr(best_num_neur))
print('  - optimal learning rate = {:.1f}'.format(best_learn_rate))
print('  - optimal tolerance = ' + repr(best_tol))
print('  - best performing randomly initialized weights')
print()
print('is {:.4f}'.format(tot_accuracy))
print('   ======')
print()
print('Time elapsed: {:.3f} s.'.format(end4-start4))
print()

plt.show()

