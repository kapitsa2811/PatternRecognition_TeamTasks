# PatternRecognition_TeamTasks

Team tasks of pattern recognition lecture at Univesity of Fribourg.

### Team 'Over9000'
- Livio Baetscher
- Carl Balmer
- Mathias Fuchs
- Manuela Haefliger
- Marc-Antoine Jacques

### Language
- Python 3.6.0

## Exercise 2a

### Packages
- csv
- time
- numpy 1.12.0
- scikit-learn 0.18.1

### Data
- train.csv
- test.csv

### Description
The script `svm.py` builds two SVMs with the provided training set `train.csv` and applies the trained SVMs to classify
the test set `test.csv`. Two different kernels (linear and RBF) are investigated and the SVM parameters are optimized
by means of cross-validation (5-fold). The script will output the average accuracy during cross-validation for the
investigated kernels and all parameter values (C for linear / C and  &gamma; for RBF). The script will also output
the accuracy on the whole test set with the optimized parameter values.

### Instructions
The script can be run in once. For that, the two files `train.csv`and `test.csv`
have to be in the same folder as the script `svm.py`. Note: the running time is quite
immense for the entire data set! You might only include a part of the data set by uncommenting the lines
```python
trainFeatures = trainFeatures[:1000, :]
trainLabels = trainLabels[:1000]
testFeatures = testFeatures[:100, :]
testLabels = testLabels[:100]
```
(and maybe adjust the number of samples to include) and/or adjust the k-fold by changing the variables `cv_num_lin` and `cv_num_rbf`
as well as the tested kernel-parameters `params_lin` and `params_rbf`.

### Results

From running the script with the entire data set (26'999 samples in `train.csv`,
15'001 samples in `test.csv`).

#### Linear kernel

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

#### RBF kernel

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
