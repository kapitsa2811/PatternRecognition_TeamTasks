"""
Team: Over9000
Authors:
- Livio Baetscher
- Carl Balmer
- Mathias Fuchs
- Manuela Haefliger
- Marc-Antoine Jacques
Date created: 20/3/2017
Date last modified: 20/3/2017
Python Version: 3.6

Pattern Recognition - Exercise 2a
"""

import csv
import numpy as np
import scipy.stats
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV

with open('Exercise_2a/data/train.csv', 'r') as f:
    reader = csv.reader(f)
    train = list(reader)

with open('Exercise_2a/data/test.csv', 'r') as f:
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

# We no longer need the samples with labels and the objects for import
del train, test, reader, f


# ---------------------------------------------------------------------------------------------------------------------
# settings

num_of_iter = 10
num_of_cv_sets = 5
dist_param_C = 30
dist_param_gamma = 0.1

# ---------------------------------------------------------------------------------------------------------------------
# linear kernel

# train svc with cross validation (divided in 'num_of_cv_sets' sets) for 'num_of_iter' random parameters C
# which are chosen from distribution exp(dist_param_C)
svc_linear = svm.SVC(kernel='linear')
random_search_linear = RandomizedSearchCV(svc_linear, {'C': scipy.stats.expon(scale=dist_param_C)}, cv=num_of_cv_sets, scoring='accuracy', n_iter=num_of_iter)
random_search_linear.fit(trainFeatures, trainLabels)
# read out the chosen values for C
C_chosen_linear = np.array(random_search_linear.cv_results_['param_C'], dtype=float)
# read out the mean accuracy for the cross-validation
mean_acc_linear = random_search_linear.cv_results_['mean_test_score']
best_C_linear = random_search_linear.best_params_
# train with chosen C
svc_best_linear = svm.SVC(kernel='linear', C=best_C_linear)
svc_best_linear.fit(trainFeatures, trainLabels)
# predict test set
predicted_labels_linear = svc_best_linear.predict(testFeatures)
# calculate accuracy
accuracy_linear = np.sum(np.equal(predicted_labels_linear, testLabels))/len(testLabels)

print(C_chosen_linear)
print(mean_acc_linear)
print(best_C_linear)
print(accuracy_linear)


# ---------------------------------------------------------------------------------------------------------------------
# RBF kernel

# train svc with cross validation (divided in 'num_of_cv_sets' sets) for 'num_of_iter' random parameters C and gamma,
# which are chosen from distributions exp(dist_param_C) and exp(dist_param_gamma)
svc_rbf = svm.SVC(kernel='rbf')
random_search_rbf = RandomizedSearchCV(svc_rbf, {'C': scipy.stats.expon(scale=dist_param_C), 'gamma': scipy.stats.expon(scale=dist_param_gamma)}, cv=num_of_cv_sets, scoring='accuracy', n_iter=num_of_iter)
random_search_rbf.fit(trainFeatures, trainLabels)
# read out the chosen values for C and gamma
C_chosen_rbf = np.array(random_search_rbf.cv_results_['param_C'], dtype=float)
gamma_chosen_rbf = np.array(random_search_rbf.cv_results_['param_gamma'], dtype=float)
# read out the mean accuracy for the cross-validation
mean_acc_rbf = random_search_rbf.cv_results_['mean_test_score']
best_C_rbf = random_search_linear.best_params_[0]
best_gamma_rbf = random_search_linear.best_params_[1]
# train with chosen C and gamma
svc_best_rbf = svm.SVC(kernel='rbf', C=best_C_rbf, gamma=best_gamma_rbf)
svc_best_rbf.fit(trainFeatures, trainLabels)
# predict test set
predicted_labels_rbf = svc_best_rbf.predict(testFeatures)
# calculate accuracy
accuracy_rbf = np.sum(np.equal(predicted_labels_rbf, testLabels))/len(testLabels)

print(C_chosen_rbf)
print(gamma_chosen_rbf)
print(mean_acc_rbf)
print(best_C_rbf)
print(accuracy_rbf)
