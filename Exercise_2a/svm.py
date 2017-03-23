"""
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
from sklearn import svm

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

# Example for SVM with linear kernel

linear_svc = svm.SVC(kernel='linear', C=1, gamma=0.002)
linear_svc.fit(trainFeatures, trainLabels)
predicted_labels_linear = linear_svc.predict(testFeatures)
accuracy_linear = np.sum(np.equal(predicted_labels_linear, testLabels))/len(testLabels)
scores = cross_val_score(clf, testFeatures, testLabels, cv=5)
scores


# Example for SVM with RBF kernel

rbf_svc = svm.SVC(kernel='rbf', C=1, gamma=0.002)
rbf_svc.fit(trainFeatures, trainLabels)
predicted_labels_rbf = rbf_svc.predict(testFeatures)
accuracy_rbf = np.sum(np.equal(predicted_labels_rbf, testLabels))/len(testLabels)
