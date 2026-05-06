# Understanding Machine Learning: The Mathematical Structure of Least-Squares Support Vector Machines

As part of my 3rd Year of my Mathematics BSc Hons, I was able to do a BSc Project
in any mathematical field of my choosing. I chose to dive deeper into the mathematics of
machine learning, and specifically, least-square support vector machines (LS-SVMs).

## Overview

- Derivation of the LS-SVM, from knowledge of linear regression, and basic linear algebra
- Introducing an iterative weighting for our datapoints to find models that are better at prediction
- Using a One-vs-Rest (OvR) multi-class classification method for the MNIST dataset
- Using subsets of the MNIST dataset to see how the model performs with limited training data:

|         | No. of data points | Accuracy |
|---------|--------------------|----------|
| Model 1 | 1,000              | 93.41%   |
| Model 2 | 2,000              | 94.69%   |
| Model 3 | 5,000              | 96.84%   |
| Model 4 | 7,500              | 97.28%   |
| Model 5 | 10,000             | 97.67%   |
| Model 6 | 15,000             | 98.00%   |
| Model 7 | 30,000             | 98.40%   |
| Model 8 | 60,000             | 98.76%   |

## Structure

- `code/` — training and evaluation code files
- `assets/` — MNIST data files and saved models

## Requirements

Python 3.x, NumPy
