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

### Results

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

Accuracy of SVM with optimal parameter C = and &gamma; = on training set is .
