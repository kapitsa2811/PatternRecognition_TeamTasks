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

## Exercise 2a - SVM

### Packages
- csv
- time
- numpy 1.12.0
- scikit-learn 0.18.1
- optunity 1.1.1 (for installation see [here](http://optunity.readthedocs.io/en/latest/user/installation.html))

### Data
- train.csv
- test.csv

### Description
The scripts `svmGridsearch.py` and `svmOptunity.py` build two SVMs with the provided training set `train.csv` each and apply the trained SVMs to classify
the test set `test.csv`. The two SVMs use different kernels (linear and RBF). Different SVM parameters are optimized
by means of cross-validation (5-fold). For that, the script `svmGridsearch.py` uses the `GridSearchCV` method from the module `sklearn.model_selection`
and the script `svmOptunity.py` uses the module `optunity`.
The scripts will output the average accuracy during cross-validation of the
investigated kernels and all parameter values (C for linear / C and  &gamma; for RBF). The scripts will also output
the accuracy on the whole test set with the optimized parameter values.

### Instructions
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

### Results
From running the scripts with the entire data set (26'999 samples in `train.csv`,
15'001 samples in `test.csv`).

#### Linear kernel - gridsearch
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

#### Linear kernel - optunity
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

#### RBF kernel - gridsearch
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

#### RBF kernel - optunity
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

### Conclusion
The RBF kernel performs about 4.7% better than the linear kernel. The parameter C doesn't matter much in the linear kernel. It is best to choose the default parameter C=1. For the RBF kernel, the choice of the parameters is much more crucial. The tests have shown, that C=3 or 4 and &gamma;=0.01 or 0.001,  or any value in between, yields the best results.

## Exercise 2b - MLP

### Packages
- csv
- time
- numpy 1.12.0
- scikit-learn 0.18.1


### Data
- train.csv
- test.csv

### Description
The script `mlp.py` trains an MLP with one hidden layer from the training set `train.csv` and applies the trained MLP to classify the test set `test.csv`.

Before doing so, a first cross-validation (5-fold) is done for the parameters
- number of neurons in the hidden layer (in the range [10, 100] with step size 10)
- learning rate used in stochastic gradient descent (in the range [0.1, 1] with step size 0.1)

In a second step, the number of training iterations (combined with the convergence tolerance) are investigated and optimized during cross-validation (5-fold).
For that, a graph showing the loss on the training set and the validation sets with respect to the training epochs is plotted.

In a third step, another cross-validation (5-fold) is performed for testing several different random initializations for the neuron weights.

The script will output the results of the three steps as well as the total accuracy on the test set with the best parameters found during the above mentioned steps.

### Instructions
As in exercise 2a, the script, here `mlp.py`, located in the folder `Exercise_2b`, can be run at once. Several adjustments can be done in the settings part of the script.

### Results
The full output of the script (run with the entire data set) can be found in the file `Exercise_2b\output.txt`.

#### First cross-validation (for number of neurons and learning rate)
The top ten parameter combinations regarding accuracy during the first cross-validation are:

number of neurons | learning rate | accuracy
--- | --- | ---
100 | 0.2 | 0.9698
 90 | 0.2 | 0.9698
100 | 0.1 | 0.9685
 90 | 0.1 | 0.9684
 90 | 0.3 | 0.9681
 80 | 0.1 | 0.9677
 80 | 0.2 | 0.9676
 90 | 0.4 | 0.9651
100 | 0.3 | 0.9643
100 | 0.4 | 0.9643

So, the best choice is 100 neurons and a learning rate of 0.2.

#### Second cross-validation (for tolerance)
Notes on how convergence in the `MLPClassifier` works:

a tolerance value `tol` can be set as parameter:

> **tol** : float, optional, default 1e-4
>
> Tolerance for the optimization. When the loss or score is not improving by at least tol for two consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.

as well as the maximum value of iterations `max_iter`:

> **max_iter** : int, optional, default 200
>
> Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.

While working with the `MLPClassifier`, handling the tolerance value turned out to be more effective than trying to control the number of iterations.

So, the following plots show the loss function vs training epochs for the investigated tolerance values during cross validation and training:

![alt text](https://github.com/nela3003/PatternRecognition_TeamTasks/blob/master/Exercise_2b/plot_tolerance.png "Loss Function")

From the plots, it can be observed that a tolerance of 0.0001 and 0.00001 doesn't help much in decreasing the loss, since already after 15
iterations the gain of further training is negligible. A choice of 0.1 or 0.01 for the tolerance is maybe to optimistic, since there could still be some improvement done.
So, a choice of 0.001 for the tolerance as convergence criteria makes sense.

#### Third cross-validation (for neuron weights due to different random initialization)

Randomly initializing the weights several times yields to the following accuracy values:

random state | accuracy
---- | ----
  6 | 0.9670
 39 | 0.9657
 97 | 0.9665
 78 | 0.9685
 56 | 0.9303
 10 | 0.9629
 62 | 0.9654
 40 | 0.9257
 96 | 0.9704
 74 | 0.9686

hence an improvement of about 4% can be observed.

#### Predict labels for test set and compute accuracy
Accuracy of MLP on the entire training set with
  - optimal number of neurons = 100
  - optimal learning rate = 0.2
  - optimal tolerance = 0.001
  - best performing randomly initialized weights

is 0.9716.

### Conclusion
MLPs are much faster than SVMs. Running the whole script can be done in about half an hour although many more MLPs are trained than in Exercise 2a. Regarding accuracy is MLP in the same range as SVM.
Increasing the number of neurons doesn't increase the running time notably, but increasing the learning rate does slightly.