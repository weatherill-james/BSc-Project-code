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
  - **kernel_ridge_regression.py** - Implementation of KRR using made-up data
  - **ls_svm_bullseye.py** - Implementation of LS-SVM on made-up bullseye dataset
  - **ls_svm_messy_data.py** - LS-SVM on made-up data with extreme outliers (to compare to wLS-SVM on same data)
  - **w_ls_svm_messy_data.py** - wLS-SVM on made-up data with extreme outliers (to compare to LS-SVM on same data)
  - **w_ls_SVM_RBF_MNIST.py** - Multi-class RBF wLS-SVM on tunable subsets of the MNIST dataset
  - **w_ls_SVM_RBF_MNIST_full.py** - Multi-class RBF wLS-SVM on the full MNIST training set
  - **w_ls_SVM_RBF_MNIST_no_training.py** - A file to run different saved models of the multi-class RBF wLS-SVM for different sized training sets
  - **rendering_MNIST_pdfs.py** - Rendering some MNIST dataset images using Matplotlib
- `assets/` — MNIST data files and saved models
  - **MNIST_testing_data_10k.gz** - Lossless compressed gzip file of the 10,000 MNIST testing points
  - **MNIST_testing_labels_10k.gz** - gzip file for the 10,000 MNIST testing points' class labels
  - **MNIST_training_data_60k.gz** - gzip file for the 60,000 MNIST training points
  - **MNIST_training_labels_60k.gz** - gzip file for the 60,000 MNIST training points' class labels
  - **wlssvm_model_1k.npz** - Saved NumPy arrays, created by training on 1,000 MNIST training points
  - **wlssvm_model_2k.npz** - Saved NumPy arrays, 2,000 MNIST training points
  - **wlssvm_model_5k.npz** - Saved NumPy arrays, 5,000 MNIST training points
  - **wlssvm_model_7o5k.npz** - Saved NumPy arrays, 7,500 MNIST training points
  - **wlssvm_model_10k.npz** - Saved NumPy arrays, 10,000 MNIST training points
  - **wlssvm_model_15k.npz** - Saved NumPy arrays, 15,000 MNIST training points
  - **wlssvm_model_30k.npz** - Saved NumPy arrays, 30,000 MNIST training points
  - **wlssvm_model_60k.npz** - Saved NumPy arrays, the full 60,000 MNIST training points

## Requirements

Python 3.x, NumPy

## Installation

<ins>Clone</ins> this repository (do not use "Download ZIP", since some `.npz` files are large and are committed using Git's "large file storage" (LFS). "Download ZIP" will not download them correctly):
```bash
git clone https://github.com/James-Weatherill/BSc-Project-code.git
cd BSc-Project-code
git lfs install
git lfs pull
```
