# PatternRecognition_TeamTasks

Team tasks to lecture "Pattern Recognition" at University of Fribourg.

### Team 'Over9000'
- Livio Baetscher
- Carl Balmer
- Mathias Fuchs
- Manuela Haefliger
- Marc-Antoine Jacques

### Language
- Python 3.6.1

## Final Validation

The final validation of the three Exercises (MNIST, KWS and Molecules) has been performed and the results can be found in the corresponding folders.

### Exercise 2 - MNIST

For this task, the MLP has been chosen over the SVM. The final validation of exercise 2 has been done in the file `Exercise_2b/validation.py` and the results can be found in the file `Exercise_2b/output.txt`. As training set, the same file is used as the one used during the exercise (`train.csv`, containig 26'999 samples). As test set, a new file, `mnist_test.csv`, containing 10'000 samples, was used. The parameters were chosen as follows:
- number of hidden layer = 1
- number of neurons = 100
- activation function for the hidden layers = rectified linear unit function
- solver for weight optimization = stochastic gradient descent
- learning rate = 0.1 (constant)
- maximum number iterations = 200
- tolerance for optimization = 0.0001

The output file with the results is formatted like

      test_ID1 predicted_class1
      test_ID2 predicted_class2
      test_ID3 predicted_class3
      ...

where `test_IDx` refers to row x of the file `mnist_test.csv` and `predicted_classx` refers to the class prediction (0, 1, ... , 9) for the x-th row in the file `mnist_test.csv`.

### Exercise 3 - KWS

The final validation of exercise 3 has been done in the file `Exercise_3/Evaluation.py` and the results can be found in the file `Exercise_3/results.txt`. As training set, the 1'177 provided words were used and can be found in the folder `Exercise_3/data/cropped_words/`. The test set contains 10 words and they're listed in the file `Exercise_3/data/validation/keywords.txt`.

The output file with the results is formatted like

      Keyword1, testword_ID1, dissimilarity1, testword_ID2, dissimilarity2, ...
      Keyword2, testword_ID1, dissimilarity1, testword_ID2, dissimilarity2, ...
      Keyword2, testword_ID1, dissimilarity1, testword_ID2, dissimilarity2, ...
      ...

### Exercise 4 - Molecules

The final validation of exercise 4 has been done in the file `Exercise_4/validation.py` and the results can be found in the file `Exercise_4/output.txt`. As training set, the same 250 molecules are used as in the exercise itself (can be found in `Exercise_4/data/glx/` according to the entries in the file `Exercise_4/data/train.txt`). The test set contains 1'500 molecules, located in the folder `Exercise_4/data/test/`. For the costs used in the Dirac cost function, `Cn = Ce = 1` was taken and for kNN, `k = 3`was chosen.

The output file with the results is formatted like

      test_ID1 predicted_class1
      test_ID2 predicted_class2
      test_ID3 predicted_class3
      ...

where `test_IDx` refers to the molecule saved in the file `Exercise/data/test/test_IDx.glx` and `predicted_classx` refers to the class prediction (a or i) for the molecule saved in the file `Exercise/data/test/test_IDx.glx`.

## Content

Each exercise has its own folder and its own README file.

* [Exercise 2a - SVM](Exercise_2a)
    * [Packages](Exercise_2a#packages)
    * [Data](Exercise_2a#data)
    * [Description](Exercise_2a#description)
    * [Instructions](Exercise_2a#instructions)
    * [Results](Exercise_2a#results)
        * [Linear kernel - gridsearch](Exercise_2a#linear-kernel---gridsearch)
        * [Linear kernel - optunity](Exercise_2a#linear-kernel---optunity)
        * [RBF kernel - gridsearch](Exercise_2a#rbf-kernel---gridsearch)
        * [RBF kernel - optunity](Exercise_2a#rbf-kernel---optunity)
    * [Conclusion](Exercise_2a#conclusion)

* [Exercise 2b - MLP](Exercise_2b)
    * [Packages](Exercise_2b#packages)
    * [Data](Exercise_2b#data)
    * [Description](Exercise_2b#description)
    * [Instructions](Exercise_2b#instructions)
    * [Results](Exercise_2b#results)
        * [First cross-validation (for number of neurons and learning rate)](Exercise_2b#first-cross-validation-for-number-of-neurons-and-learning-rate)
        * [Second cross-validation (for tolerance)](Exercise_2b#second-cross-validation-for-tolerance)
        * [Third cross-validation (for neuron weights due to different random initialization)](Exercise_2b#third-cross-validation-for-neuron-weights-due-to-different-random-initialization)
        * [Predict labels for test set and compute accuracy](Exercise_2b#predict-labels-for-test-set-and-compute-accuracy)
    * [Conclusion](Exercise_2b#conclusion)

* [Exercise 3 - Keyword Spotting](Exercise_3)
    * [Packages](Exercise_3#packages)
    * [Data](Exercise_3#data)
    * [Description](Exercise_3#description)
    * [Instructions](Exercise_3#instructions)
    * [Results](Exercise_3#results)
    * [Conclusion](Exercise_3#conclusion)

* [Exercise 4 - Molecules](Exercise_4)
    * [Packages](Exercise_4#packages)
    * [Data](Exercise_4#data)
    * [Description](Exercise_4#description)
    * [Calculating cost matrix C](Exercise_4#calculating-cost-matrix-c)
    * [Instructions](Exercise_4#instructions)
    * [Results](Exercise_4#results)
    * [Conclusion](Exercise_4#conclusion)
