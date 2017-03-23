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

with open('train100.csv', 'r') as f:
    reader = csv.reader(f)
    train = list(reader)

with open('test10.csv', 'r') as f:
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

# We no longer need the samples with the labels
del train, test, reader, f

# Example for SVM with linear kernel

linear_svc = svm.SVC(kernel='linear', C=1, gamma=0.002)
linear_svc.fit(trainFeatures, trainLabels)
linear_svc.predict(testFeatures)

# Example for SVM with RBF kernel

rbf_svc = svm.SVC(kernel='rbf', C=1, gamma=0.002)
rbf_svc.fit(trainFeatures, trainLabels)
rbf_svc.predict(testFeatures)

