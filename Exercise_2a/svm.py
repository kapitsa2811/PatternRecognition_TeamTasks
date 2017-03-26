"""
Team: Over9000

Python Version: 3.6

Pattern Recognition - Exercise 2a
"""

import csv, time
import numpy as np
from sklearn.svm import SVC
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

# trainFeatures = trainFeatures[:10000, :]
# trainLabels = trainLabels[:10000]
# testFeatures = testFeatures[:1000, :]
# testLabels = testLabels[:1000]

# ---------------------------------------------------------------------------------------------------------------------
# LINEAR KERNEL

start_lin = time.time()

# number of sets in cross validation:
cv_num_lin = 5

# parameters to test in grid search:
params_lin = {"C": [1, 2, 3, 4, 5, 6, 7, 8, 9]}

print()
print("---- LINEAR KERNEL ----")

# set up svm for parameter optimization
svm_lin = SVC(kernel='linear', cache_size=4000)

# grid search
gs_svm_lin = GridSearchCV(svm_lin, param_grid=params_lin, scoring='accuracy', n_jobs=2, cv=cv_num_lin)
gs_svm_lin.fit(trainFeatures, trainLabels)
# note: in the end, the svm is refit with the best estimator and the entire data set

# read tested parameter combinations and their scores
params_gs_lin = gs_svm_lin.cv_results_['params']
score_cv_lin = gs_svm_lin.cv_results_['mean_test_score']

# output tested parameters and scores
print()
print('C | accuracy')
print('------------')
for i in range(len(params_gs_lin)):
    print('{:1.0f} | {:1.4f}'.format(params_gs_lin[i]['C'], score_cv_lin[i]))
print()

# predict labels for test set
pred_labels_lin = gs_svm_lin.predict(testFeatures)
accuracy_lin = metrics.accuracy_score(pred_labels_lin, testLabels)

end_lin = time.time()

best_C_lin = gs_svm_lin.best_params_['C']

print("Accuracy of SVM with linear kernel and optimal parameter C = {:1.0f} on training set is {:1.4f}".format(best_C_lin, accuracy_lin))
print('Time elapsed for linear kernel: ' + repr(end_lin-start_lin) + ' s')
print()

# ---------------------------------------------------------------------------------------------------------------------
# RBF KERNEL

start_rbf = time.time()

# number of sets in cross validation:
cv_num_rbf = 5

# parameters to test in grid search:
params_rbf = {"C": [1, 2, 3, 4], "gamma": [0.01, 0.001, 0.0001]}

print()
print("---- RBF KERNEL ----")

# set up svm for parameter optimization
svm_rbf = SVC(kernel='rbf', cache_size=4000)

# grid search
gs_svm_rbf = GridSearchCV(svm_rbf, param_grid=params_rbf, scoring='accuracy', n_jobs=2, cv=cv_num_rbf)
gs_svm_rbf.fit(trainFeatures, trainLabels)
# note: in the end, the svm is refit with the best estimator and the entire data set

# read tested parameter combinations and their scores
params_gs_rbf = gs_svm_rbf.cv_results_['params']
score_cv_rbf = gs_svm_rbf.cv_results_['mean_test_score']

# output tested parameters and scores
print()
print('C | gamma   | accuracy')
print('----------------------')
for i in range(len(params_gs_rbf)):
    print('{:1.0f} | {:1.5f} | {:1.4f}'.format(params_gs_rbf[i]['C'], params_gs_rbf[i]['gamma'], score_cv_rbf[i]))
print()

# predict labels for test set
pred_labels_rbf = gs_svm_rbf.predict(testFeatures)
accuracy_rbf = metrics.accuracy_score(pred_labels_rbf, testLabels)

end_rbf = time.time()

best_C_rbf = gs_svm_rbf.best_params_['C']
best_gamma_rbf = gs_svm_rbf.best_params_['gamma']

print("Accuracy of SVM with RBF kernel and optimal parameter C = {:1.0f} and gamma = {:1.5f} on training set is {:1.4f}".format(best_C_rbf, best_gamma_rbf, accuracy_rbf))
print('Time elapsed for RBF kernel: ' + repr(end_rbf-start_rbf) + ' s')
print()
