# PatternRecognition_TeamTasks

### Authors:
- Livio Baetscher <3
- Carl Balmer
- Mathias Fuchs
- Manuela Haefliger
- Marc-Antoine Jacques

### Language
- Python 3.6.0

## Exercise 2a

### Packages
- csv
- numpy 1.12.0
- scikit-learn 0.18.1

### Description
This script builds a SVM with the provided training set. Then it applies the trained SVM to classify the test set. Two
different kernels (linear and RBF) are investigated and the SVM parameters are optimized by means of cross-validation.
The script will output the average accuracy during cross-validation for the investigated kernels and all parameter
values (C and  &gamma;), as well as the accuracy on the test set with the optimized parameter values.