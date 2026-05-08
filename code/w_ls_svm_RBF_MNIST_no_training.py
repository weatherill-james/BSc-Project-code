import os
import gzip
import numpy as np
from time import time

start_time = time()

########################################

# Functions

def parse_data(path_name):
    with gzip.open(path_name, "rb") as file:
        raw_file = file.read()
    return np.frombuffer(raw_file, dtype = np.uint8, offset = 16).reshape(-1, 784).astype(np.float64) / 255.0

def parse_labels(path_name):
    with gzip.open(path_name, "rb") as file:
        raw_file = file.read()
    return np.frombuffer(raw_file, dtype = np.uint8, offset = 8).astype(int)

def k_func(new_X, data_X, gamma):
    new_q  = np.sum(new_X ** 2, axis = 1)
    data_q = np.sum(data_X ** 2, axis = 1)
    return np.exp(gamma * (2 * (new_X @ data_X.T) - new_q[:, None] - data_q[None, :]))

########################################

assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# We load our npz file, which contains the training data, alphas, w0s, and gamma value. This is everything we need to make predictions
# Change the filename here (e.g. to "wlssvm_model_15k.npz") to load a different model
model = np.load(os.path.join(assets_path, "wlssvm_model_60k.npz"))

# This is how we extract the data from the npz file
X_training = model["X_training"]
alphas     = model["alphas"]
w0s        = model["w0s"]
# Gamma is a 1x1 array, so '.item()' just extracts the value from the array
gamma      = model["gamma"].item()

X_testing  = parse_data(os.path.join(assets_path, "MNIST_testing_data_10k.gz"))
y_testing  = parse_labels(os.path.join(assets_path, "MNIST_testing_labels_10k.gz"))

q_testing  = np.sum(X_testing ** 2, axis = 1)
K_testing = k_func(X_testing, X_training, gamma)

scores      = np.array([K_testing @ alphas[i] + w0s[i] for i in range(10)])
predictions = np.argmax(scores, axis = 0)
accuracy    = np.mean(predictions == y_testing)
print(f"\nTest accuracy: {round(accuracy * 100, 3)}%")

correct   = np.sum(predictions == y_testing)
incorrect = np.sum(predictions != y_testing)

print(f"\n  Correct predictions:  {correct} / {len(y_testing)}")
print(f"Incorrect predictions: {incorrect} / {len(y_testing)}")

########################################

finish_time = time()

print(f"\nThe code took: {finish_time - start_time} seconds to run\n")
