"""
Team: Over9000

Python Version: 3.6

Pattern Recognition - Exercise 2a
"""

import csv
import numpy as np
import sklearn.svm
import optunity.metrics

# ---------------------------------------------------------------------------------------------------------------------
# DATA PREPARATION

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

# ---------------------------------------------------------------------------------------------------------------------
# IF YOU WANT TO USE ONLY PART OF THE DATA

# trainFeatures = trainFeatures[:1000, :]
# trainLabels = trainLabels[:1000]
# testFeatures = testFeatures[:100, :]
# testLabels = testLabels[:100]

# ---------------------------------------------------------------------------------------------------------------------
# LINEAR KERNEL

# number of sets in cross validation:
cv_num_lin = 5

# parameters to test in grid search:
param_C_lin = [0, 5]
num_params_lin = 6

# Optunity cross validation on training set
@optunity.cross_validated(x=trainFeatures.tolist(), y=trainLabels.tolist(), num_folds=cv_num_lin)
def svm_linear_tuned_acc(x_train, y_train, x_test, y_test, C):
    model = sklearn.svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = np.sum(np.equal(y_test, y_pred))/len(y_test)
    return acc
optimal_linear_pars, info_linear, _ = optunity.maximize(svm_linear_tuned_acc, num_evals=num_params_lin, C=param_C_lin)

print()
print("--- LINEAR KERNEL ---")
print("Optimal parameters: " + str(optimal_linear_pars))
print("Accuracy of tuned SVM with linear kernel: {:1.3f}".format(info_linear.optimum))

# output tested parameters and scores
df_linear = optunity.call_log2dataframe(info_linear.call_log)
print(df_linear)

# train svm with optimal parameter and calculate accuracy
svc_best_linear = sklearn.svm.SVC(kernel='linear', C=optimal_linear_pars['C'])
svc_best_linear.fit(trainFeatures, trainLabels)
predicted_labels_linear = svc_best_linear.predict(testFeatures)
accuracy_linear = float(np.sum(np.equal(predicted_labels_linear, testLabels))/len(testLabels))

print("Accuracy of SVM with linear kernel and optimal parameter C = {:1.3f} on training set is {:1.3f}".format(optimal_linear_pars['C'], accuracy_linear))
print()

# ---------------------------------------------------------------------------------------------------------------------
# RBF kernel

# number of sets in cross validation:
cv_num_rbf = 5

# parameters to test in grid search:
param_C_rbf = [1, 5]
param_gamma_rbf = [0.0006, 0.006]
num_params_rbf = 6

# Optunity cross validation on training set
@optunity.cross_validated(x=trainFeatures.tolist(), y=trainLabels.tolist(), num_folds=cv_num_rbf)
def svm_rbf_tuned_acc(x_train, y_train, x_test, y_test, C, gamma):
    model = sklearn.svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = np.sum(np.equal(y_test, y_pred))/len(y_test)
    return acc
optimal_rbf_pars, info_rbf, _ = optunity.maximize(svm_rbf_tuned_acc, num_evals=num_params_rbf, C=param_C_rbf, gamma=param_gamma_rbf)

print()
print("--- RBF KERNEL ---")
print("Optimal parameters: " + str(optimal_rbf_pars))
print("Accuracy of tuned SVM with RBF kernel: {:1.3f}".format(info_rbf.optimum))

# output tested parameters and scores

df_rbf = optunity.call_log2dataframe(info_rbf.call_log)
print(df_rbf)

# train svm with optimal parameter and calculate accuracy
svc_best_rbf = sklearn.svm.SVC(kernel='rbf', C=optimal_rbf_pars['C'], gamma=optimal_rbf_pars['gamma'])
svc_best_rbf.fit(trainFeatures, trainLabels)
predicted_labels_rbf = svc_best_rbf.predict(testFeatures)
accuracy_rbf = float(np.sum(np.equal(predicted_labels_rbf, testLabels))/len(testLabels))

print("Accuracy of SVM with RBF kernel and optimal parameter C = {:1.3f} and gamma = {:1.3f} on training set is {:1.3f}".format(optimal_rbf_pars['C'], optimal_rbf_pars['gamma'], accuracy_rbf))
print()
