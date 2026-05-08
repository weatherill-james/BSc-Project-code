import os
import gzip
import numpy as np
from time import time

start_time = time()

########################################

# Functions

# These functions are capable of parsing, and harvesting the raw binary data from the MNIST files
## 'frombuffer' is a NumPy function that reads raw binary bytes into a numpy array
## 'dtype' specifies that each byte is one unsigned 8-bit integer
## 'offset' makes NumPy skip the first 16 bytes, which are just file header information
## 'reshape' tells NumPy that the data should be reshaped into an unspecified number of rows (the -1), and 784 columns (the number of pixels in each image), so that each image becomes just one row of 784 values
## 'astype' converts the integers into 64-bit floats, to prevent NumPy from truncating our values when we divide by 255
## We divide by 255 to normalise our pixel values to be between 0 and 1, which keeps our kernel values from getting too large, which can cause numerical instability
def parse_data(path_name):
    with gzip.open(path_name, "rb") as file:
        raw_file = file.read()
    return np.frombuffer(raw_file, dtype = np.uint8, offset = 16).reshape(-1, 784).astype(np.float64) / 255.0

def parse_labels(path_name):
    with gzip.open(path_name, "rb") as file:
        raw_file = file.read()
    return np.frombuffer(raw_file, dtype = np.uint8, offset = 8).astype(int)

# New k_func that computes the RBF kernel, instead of a polynomial kernel
def k_func(new_X, data_X, gamma):
    # Finding |p_{i}|^{2} for each point, and framing it as a column vector
    new_q  = np.sum(new_X ** 2, axis = 1)
    data_q = np.sum(data_X ** 2, axis = 1)
    return np.exp(gamma * (2 * (new_X @ data_X.T) - new_q[:, None] - data_q[None, :]))

def new_v_inv(matrix, alpha, w_0, y_vec, delta):
    errors = (matrix @ alpha + w_0) - y_vec
    return np.diag((errors ** 2) + delta)

########################################

# Training the wLS-SVM model

# This is the path to the "assets" folder, which is where our MNIST files are
assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# Number of training points
n = 15_000

# Setting our parameters for our wLS-SVM model
gamma = 0.0185
lamb = 0.0001
delta = 1e-6
epsilon = 1e-10

# One tenth of the training points per class
point_per_class = n // 10

# Loading the full MNIST training dataset, which we will subset
# The first file contains the training data in a specific order
X_training = parse_data(os.path.join(assets_path, "MNIST_training_data_60k.gz"))
# This second file contains the training labels, in the same order, so we can pair them up
y_training = parse_labels(os.path.join(assets_path, "MNIST_training_labels_60k.gz"))

# We choose a seed (3), to ensure we get the SAME random subset of training data each time we run the code
rng = np.random.default_rng(3)
# This selects 1,500 random training point indices for each class, from 0 to 9, and concatenates them together
selected_indices = np.concatenate([rng.choice(np.where(y_training == point_class)[0], size = point_per_class, replace = False) for point_class in range(10)])
# Not strictly necessary, but ensures our training points won't be grouped by class. The results would be identical without this line
rng.shuffle(selected_indices)

# Putting our training data into our X matrix
X = X_training[selected_indices]
# Using our kernel function, above, to compute our training-data kernel matrix
K = k_func(X, X, gamma)

# Vector of ones to be used in the bordered matrix
ones_vec = np.ones((n, 1))

# Since we are doing multi-class classification, we will have 10 separate alpha vectors and w0's, which we need to store. These dictionaries can keep both vectors and scalars, and just use the digit as the key
alphas = {}
w0s    = {}

# We begin training our wLS-SVM model by looping through each digit, treating that digit as the positive class
for digit in range(10):
    print(f"\nTraining wLS-SVM for digit {digit}...")
    # We assign our class labels for all digits
    y_binary  = np.where(y_training[selected_indices] == digit, 1, -1)
    y         = np.append(y_binary, 0)
    V_inv     = np.eye(n)
    # We are using a while loop, so we create a counter to keep track of what iteration we are on
    iteration = 0

    # Using a while loop which breaks when V_inv is sufficiently close to V_inv_new (convergence)
    while True:
        iteration += 1
        K_regularized = K + lamb * V_inv
        LHS_matrix    = np.block([[K_regularized, ones_vec],
                                  [ones_vec.T,        0   ]])
        solution      = np.linalg.solve(LHS_matrix, y)
        alpha_vec     = solution[:-1]
        w0_scalar     = solution[-1]

        V_inv_new = new_v_inv(K, alpha_vec, w0_scalar, y[:-1], delta)
        # We calculate the maximum RELATIVE change in the diagonal of V_inv, which is more effective than the absolute change
        change = np.max(np.abs(np.diag(V_inv_new) - np.diag(V_inv)) / (np.diag(V_inv) + epsilon))
        print(f"  Iteration {iteration:>3} | max V change: {change:.2e}")

        # Updating V_inv
        V_inv = V_inv_new

        # If our change is sufficiently small, we have effectively converged, and we break the loop for this digit
        if change < epsilon:
            print(f"  Converged after {iteration} iterations.")
            break

    # Adding our new alpha vector and w0 value to the dictionaries
    alphas[digit] = alpha_vec
    w0s[digit]    = w0_scalar

########################################

# We save the values computed during training, which we can load into a file and use for predictions without needing to retrain
# Only uncomment when wanting to save a new model, and ensure to update the filename

# Saving each alpha vector and w0 value into a matrix and vector, respectively, so NumPy can save them (NumPy can ONLY save arrays)
alphas_matrix = np.array([alphas[i] for i in range(10)])
w0s_vector    = np.array([w0s[i] for i in range(10)])

np.savez(os.path.join(assets_path, "wlssvm_model_2k.npz"),
         X_training = X,
         alphas     = alphas_matrix,
         w0s        = w0s_vector,
         # Converting gamma into a 1-element array, so that NumPy can save it
         gamma      = np.array([gamma]))

########################################

# Testing the wLS-SVM model on the full MNIST testing dataset

# We now load the FULL MNIST testing dataset, which consists of 10,000 testing points
X_testing  = parse_data(os.path.join(assets_path, "MNIST_testing_data_10k.gz"))
y_testing  = parse_labels(os.path.join(assets_path, "MNIST_testing_labels_10k.gz"))

# We just compute K the same way as we did for the training data, but using the testing data instead
K_testing = k_func(X_testing, X, gamma)

# The 'scores' are the values that the our prediction function outputs for each testing point for each class
scores      = np.array([K_testing @ alphas[i] + w0s[i] for i in range(10)])
# The class with the highest 'score' is what becomes our prediction for that testing point. We use 'argmax', with no adjustment, because the digits and indices line up (0 is index 0, 1 is index 1, etc.)
predictions = np.argmax(scores, axis = 0)
# 'predictions == y_testing' creates a boolean array, where '1' means a correct prediction, and '0' means an incorrect prediction. np.mean finds the sum of this array, and divides it by its size, which gives us the accuracy
accuracy    = np.mean(predictions == y_testing)
print(f"\nTest accuracy: {round(accuracy * 100, 3)}%")

correct   = np.sum(predictions == y_testing)
incorrect = np.sum(predictions != y_testing)

print(f"\n  Correct predictions: {correct} / {len(y_testing)}")
print(f"Incorrect predictions: {incorrect} / {len(y_testing)}")

########################################

finish_time = time()

print(f"\nThe code took: {finish_time - start_time} seconds to run\n")
