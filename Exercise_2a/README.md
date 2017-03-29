# Exercise 2a - SVM

## Packages
- csv
- time
- numpy 1.12.0
- scikit-learn 0.18.1
- optunity 1.1.1 (for installation see [here](http://optunity.readthedocs.io/en/latest/user/installation.html))

## Data
- train.csv
- test.csv

## Description
The scripts `svmGridsearch.py` and `svmOptunity.py` build two SVMs with the provided training set `train.csv` each and apply the trained SVMs to classify
the test set `test.csv`. The two SVMs use different kernels (linear and RBF). Different SVM parameters are optimized
by means of cross-validation (5-fold). For that, the script `svmGridsearch.py` uses the `GridSearchCV` method from the module `sklearn.model_selection`
and the script `svmOptunity.py` uses the module `optunity`.
The scripts will output the average accuracy during cross-validation of the
investigated kernels and all parameter values (C for linear / C and  &gamma; for RBF). The scripts will also output
the accuracy on the whole test set with the optimized parameter values.

## Instructions
The scripts can be run in once. For that, the two files `train.csv`and `test.csv` (located in the folder `Exercise_2a`)
have to be in the same folder as the scripts `svm*.py`. Note: the running time is quite
immense for the entire data set! You might only include a part of the data set by uncommenting the lines
```python
trainFeatures = trainFeatures[:1000, :]
trainLabels = trainLabels[:1000]
testFeatures = testFeatures[:100, :]
testLabels = testLabels[:100]
```
(and maybe adjust the number of samples to include) and/or adjust the k-fold by changing the variables `cv_num_lin` and `cv_num_rbf`
as well as the tested kernel-parameters `params_lin` and `params_rbf` in `svmGridsearch.py` and `param_C_lin`, `num_params_lin`, `param_C_rbf`
and `num_params_rbf` in `svmOptunity.py`.

## Results
From running the scripts with the entire data set (26'999 samples in `train.csv`,
15'001 samples in `test.csv`).

### Linear kernel - gridsearch
Mean accuracy for different parameter C during cross validation:

C | accuracy
--- | ---
1 | 0.9231
2 | 0.9188
3 | 0.9171
4 | 0.9155
5 | 0.9143
6 | 0.9144
7 | 0.9141
8 | 0.9135
9 | 0.9128

Accuracy of SVM with optimal parameter C = 1 on training set is 0.9306.

### Linear kernel - optunity
Mean accuracy for different parameter C during cross validation:

C | accuracy
--- | ---
0.0391 | 0.9080
1.2891 | 0.9080
1.9141 | 0.9080
2.5391 | 0.9080
3.7891 | 0.9080
4.4141 | 0.9080

Accuracy of SVM with optimal parameter C = 4.414 on training set is 0.9080.

### RBF kernel - gridsearch
Mean accuracy for different parameter C and &gamma; during cross validation:

C | &gamma; | accuracy
--- | --- | ---
1 | 0.01000 | 0.9662
1 | 0.00100 | 0.9238
1 | 0.00010 | 0.8828
2 | 0.01000 | 0.9711
2 | 0.00100 | 0.9320
2 | 0.00010 | 0.8988
3 | 0.01000 | 0.9731
3 | 0.00100 | 0.9354
3 | 0.00010 | 0.9063
4 | 0.01000 | 0.9736
4 | 0.00100 | 0.9372
4 | 0.00010 | 0.9101

Accuracy of SVM with optimal parameter C = 4 and &gamma; = 0.01 on training set is 0.9775.

### RBF kernel - optunity
Mean accuracy for different parameter C and &gamma; during cross validation:

C | &gamma; | accuracy
--- | --- | ---
1.234375 | 0.000959 | 0.926479
2.234375 | 0.005009 | 0.961628
2.734375 | 0.002984 | 0.952184
3.234375 | 0.003659 | 0.958851
4.234375 | 0.002309 | 0.951517
4.734375 | 0.005684 | 0.968073

Accuracy of SVM with optimal parameter C = 4.734 and gamma = 0.006 on training set is 0.973.

## Conclusion
The RBF kernel performs about 4.7% better than the linear kernel. The parameter C doesn't matter much in the linear kernel. It is best to choose the default parameter C=1. For the RBF kernel, the choice of the parameters is much more crucial. The tests have shown, that C=3 or 4 and &gamma;=0.01 or 0.001,  or any value in between, yields the best results.
