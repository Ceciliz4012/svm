# svm
Implemented a SVM algorithm for binary classification and application to both synthetic and real data.

## Dataset
1. The synthetic dataset contains 200 training examples and 1800 test examples, all generated i.i.d. from a fixed probability distribution.

2. The real dataset is a spam classification dataset, where each instance is an email message represented by 57 features and is associated with a binary label indicating whether the email is spam (+1) or non-spam (−1).1 The data set is divided into training and test sets; there are 250 training examples and 4351 test examples. The goal is to learn from the training set a classifier that can classify new email messages in the test set.

## Input parameters
This program takes as input a training set (X, y), where X is an m × d matrix (m instances, each of dimension d) 
and y is an m-dimensional vector (with yi ∈ {±1} being a binary label associated with the i-th instance in X), 
parameter C, kernel type (which can be ‘linear’, ‘polynomial’ or ‘rbf’), and kernel parameter.

## Implementation
Perform SVM of 3 kernel types (linear, polynomial, or RBF) on both datasets.
Analysis and data visualization is in the analysis.pdf file.
