# k-nearest neighbors

A k-nearest neighbor implementation for predicting income from 85 dimensional data.

The program expects 8000 entries in the training set, with an id column as the first and a label column as the last. The program runs a basic training on the data and also 4-fold cross-validation for a variety of k values. It also writes its predictions for k=99 to the test_predicted.csv file.

## What I Learned

- k-nearest neighbors algorithm
  - values of cross-validation
  - trends as k increases/decreases
  - behavior as k approaches n
- K-fold cross-validation
- numpy
  - optimized matrix operations

## How To Use
Install python and numpy locally beforehand. Otherwise, simply copy the repo, run `python knn.py`
