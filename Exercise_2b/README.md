# Exercise 2b - MLP

## Packages
- csv
- time
- numpy 1.12.0
- scikit-learn 0.18.1


## Data
- train.csv
- test.csv

## Description
The script `mlp.py` trains an MLP with one hidden layer from the training set `train.csv` and applies the trained MLP to classify the test set `test.csv`.

Before doing so, a first cross-validation (5-fold) is done for the parameters
- number of neurons in the hidden layer (in the range [10, 100] with step size 10)
- learning rate used in stochastic gradient descent (in the range [0.1, 1] with step size 0.1)

In a second step, the number of training iterations (combined with the convergence tolerance) are investigated and optimized during cross-validation (5-fold).
For that, a graph showing the loss on the training set and the validation sets with respect to the training epochs is plotted.

In a third step, another cross-validation (5-fold) is performed for testing several different random initializations for the neuron weights.

The script will output the results of the three steps as well as the total accuracy on the test set with the best parameters found during the above mentioned steps.

## Instructions
As in exercise 2a, the script (here `mlp.py`) can be run at once. Several adjustments can be done in the settings part of the script.

## Results
The full output of the script (run with the entire data set) can be found in the file `output.txt`.

### First cross-validation (for number of neurons and learning rate)
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

### Second cross-validation (for tolerance)
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

From the plots, it can be observed, that a tolerance of 0.0001 and 0.00001 has already a very low loss after 15 to 20 iterations.
A choice of 0.1 or 0.01 for the tolerance is maybe to optimistic, since there could still be some improvement done.
So, if the tolerance would be set manually, a choice of 0.001 as convergence criteria seems reasonable.

In the following table, the mean accuracy can be seen for each of the investigated tolerance values. The mean accuracy us computed by taking the mean from the five accuracy values from the cross validation and the accuracy from training on the entire training set and applying to the entire test set.

tolerance | mean accuracy
--- | ---
0.1 | 0.9631
0.01 | 0.9692
0.001 | 0.9698
0.0001 |0.9703
0.00001 | 0.9706

Since the computational time is not critical and the learning only improves (loss decreases, no overfitting),
the script chooses the tolerance which performs best regarding mean accuracy to continue with.


### Third cross-validation (for neuron weights due to different random initialization)

Randomly initializing the weights several times yields to the following accuracy values:

random state | accuracy
---- | ----
 32 | 0.9676
 26 | 0.9694
 98 | 0.9626
 37 | 0.9628
 42 | 0.9701
  0 | 0.9716
 36 | 0.9700
  3 | 0.9696
 31 | 0.9551
 18 | 0.9712
 17 | 0.9671
 49 | 0.9648
  5 | 0.9633
  8 | 0.9696
 72 | 0.9674
  9 | 0.7175
 23 | 0.9650
 60 | 0.9681
 13 | 0.9658
 12 | 0.9712

hence, if the worst case with 0.7175 is ignored, an improvement of about 1.6% can be observed.

### Predict labels for test set and compute accuracy
Accuracy of MLP on the entire training set with
  - optimal number of neurons = 100
  - optimal learning rate = 0.2
  - optimal tolerance = 0.00001
  - best performing randomly initialized weights

is **0.9737**.

## Conclusion
MLPs are much faster than SVMs. Running the whole script can be done in about half an hour although many more MLPs are trained than in Exercise 2a. Regarding accuracy is MLP in the same range as SVM.
Increasing the number of neurons doesn't increase the running time notably, but increasing the learning rate does slightly.