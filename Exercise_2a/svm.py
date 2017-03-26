"""
Team: Over9000

Date created: 20/3/2017
Date last modified: 23/3/2017

Python Version: 3.6

Pattern Recognition - Exercise 2a
"""

import csv, time
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics

with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    train = list(reader)

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

# We no longer need the samples with labels and the objects for import
del train, test, reader, f

print('done importing')

# ---------------------------------------------------------------------------------------------------------------------

# trainFeatures = trainFeatures[:1000, :]
# trainLabels = trainLabels[:1000]
# testFeatures = testFeatures[:20, :]
# testLabels = testLabels[:20]

# ---------------------------------------------------------------------------------------------------------------------
# linear kernel

# settings:
c_lin = 1

print()
print("--- LINEAR KERNEL ---")

start_svm_lin = time.time()

svc_lin = SVC(kernel='linear', C=c_lin)
svc_lin.fit(trainFeatures, trainLabels)
pred_labels_lin = svc_lin.predict(testFeatures)
accuracy_lin = metrics.accuracy_score(pred_labels_lin, testLabels)

end_svm_lin = time.time()

print("Accuracy of SVM with linear kernel and optimal parameter C = {:1.5f} on training set is {:1.5f}".format(c_lin, accuracy_lin))
print('Time elapsed: ' + repr(end_svm_lin-start_svm_lin) + ' s')


# ---------------------------------------------------------------------------------------------------------------------
# RBF kernel

# settings:
c_rbf = 3
gamma_rbf = 0.0005

print()
print("--- RBF KERNEL ---")

start_svm_rbf = time.time()

svc_rbf = SVC(kernel='rbf', C=c_rbf, gamma=gamma_rbf)
svc_rbf.fit(trainFeatures, trainLabels)
pred_labels_rbf = svc_rbf.predict(testFeatures)
accuracy_rbf = metrics.accuracy_score(pred_labels_lin, testLabels)

end_svm_rbf = time.time()

print("Accuracy of SVM with RBF kernel and optimal parameter C = {:1.5f} and gamma = {:1.5f} on training set is {:1.5f}".format(c_rbf, gamma_rbf, accuracy_rbf))
print('Time elapsed: ' + repr(end_svm_rbf-start_svm_rbf) + ' s')
print()
