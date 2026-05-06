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

def new_v_inv(matrix, alpha, w_0, y_vec, delta):
    errors = (matrix @ alpha + w_0) - y_vec
    return np.diag((errors ** 2) + delta)

########################################

# Training the wLS-SVM model

assets_path = os.path.join(os.getcwd(), "..", "assets")

n = 60_000

gamma = 0.0185
lamb = 0.0001
delta = 1e-6
epsilon = 1e-10

X_training = parse_data(os.path.join(assets_path, "MNIST_training_data_60k.gz"))
y_training = parse_labels(os.path.join(assets_path, "MNIST_training_labels_60k.gz"))

# We are using the full training dataset, so X is just X_training
# Shuffling is now completely unnecessary, since the full dataset is not ordered
X = X_training
K = k_func(X, X, gamma)

ones_vec = np.ones((n, 1))

alphas = {}
w0s    = {}

for digit in range(10):
    print(f"\nTraining wLS-SVM for digit {digit}...")
    y_binary  = np.where(y_training == digit, 1, -1)
    y         = np.append(y_binary, 0)
    V_inv     = np.eye(n)
    iteration = 0

    while True:
        iteration += 1
        K_regularized = K + lamb * V_inv
        LHS_matrix    = np.block([[K_regularized, ones_vec],
                                  [ones_vec.T,        0   ]])
        solution      = np.linalg.solve(LHS_matrix, y)
        alpha_vec     = solution[:-1]
        w0_scalar     = solution[-1]

        V_inv_new = new_v_inv(K, alpha_vec, w0_scalar, y[:-1], delta)
        change = np.max(np.abs(np.diag(V_inv_new) - np.diag(V_inv)) / (np.diag(V_inv) + epsilon))
        print(f"  Iteration {iteration:>3} | max V change: {change:.2e}")

        V_inv = V_inv_new

        if change < epsilon:
            print(f"  Converged after {iteration} iterations.")
            break

    alphas[digit] = alpha_vec
    w0s[digit]    = w0_scalar

########################################

# # Only uncomment when wanting to save a new model, and ensure to update the filename

# alphas_matrix = np.array([alphas[i] for i in range(10)])
# w0s_vector    = np.array([w0s[i] for i in range(10)])

# np.savez(os.path.join(assets_path, "wlssvm_model_60k.npz"),
#          X_training = X,
#          alphas     = alphas_matrix,
#          w0s        = w0s_vector,
#          gamma      = np.array([gamma]))

########################################

# Testing the wLS-SVM model on the full MNIST testing dataset

X_testing  = parse_data(os.path.join(assets_path, "MNIST_testing_data_10k.gz"))
y_testing  = parse_labels(os.path.join(assets_path, "MNIST_testing_labels_10k.gz"))

K_testing = k_func(X_testing, X, gamma)

scores      = np.array([K_testing @ alphas[i] + w0s[i] for i in range(10)])
predictions = np.argmax(scores, axis = 0)
accuracy    = np.mean(predictions == y_testing)
print(f"\nTest accuracy: {round(accuracy * 100, 3)}%")

correct   = np.sum(predictions == y_testing)
incorrect = np.sum(predictions != y_testing)

print(f"\n  Correct predictions: {correct} / {len(y_testing)}")
print(f"Incorrect predictions: {incorrect} / {len(y_testing)}")

########################################

finish_time = time()

print(f"\nThe code took: {finish_time - start_time} seconds to run\n")
