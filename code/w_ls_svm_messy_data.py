import numpy as np
from time import time

start_time = time()

########################################

# Functions

# Upgraded way to compute the kernel function... using matrix operations
# This side-steps the need for a prediction function, since we just compute the kernel that we need (training-training or new-training)
# This also side-steps the need for a loop, which makes this method much more efficient
k_func = lambda X_new, X_old, deg, alpha, w_0: ((1 + X_new @ X_old.T) ** deg) @ alpha + w_0

# Mathematically equivalent to the method described in the paper
def new_v_inv(matrix, alpha, w_0, y_vec, delta):
    errors = (matrix @ alpha + w_0) - y_vec
    return np.diag((errors ** 2) + delta)

########################################

# (x_1, x_2, y)
data = [(-2.5, 1.5, 1), (-3.5, 1.5, 1), (-3.5, 0.5, 1), (-2.5, 0.5, 1), (-3, 1, 1), (-6, 2, 1), (-5, 1, 1), (-4, 0, 1), (-4, 2, 1), (-6, 0, 1), (5.5, 1.5, 1), (5, 0.5, 1), (6, 1, 1), (2.5, 1.5, -1), (3.5, 1.5, -1), (3.5, 0.5, -1), (2.5, 0.5, -1), (3, 1, -1), (6, 0, -1), (5, 1, -1), (4, 0, -1), (4, 2, -1), (6, 2, -1)]
data = np.array(data)

n = len(data)

d = 2
lamb = 0.01
delta = 1e-6

X = data[:, :-1]
K = (1 + X @ X.T) ** d

y = np.append(data[:, -1], 0)

# The ones vector doesn't need to be in the loop, since it doesn't change
ones_vec = np.ones((n, 1))

# We initialise V_inv, which begins as the identity matrix
V_inv = np.eye(n)

# To ensure convergence, we perform 20 loops instead of the 5-10 average
loops = 20
for i in range(loops):
    K_regularized = K + lamb * V_inv

    LHS_matrix = np.block([[K_regularized, ones_vec],
                           [ones_vec.T,        0   ]])

    solution = np.linalg.solve(LHS_matrix, y)
    alpha_vec = solution[:-1]
    w0_scalar = solution[-1]

    if i < loops - 1:
        V_inv = new_v_inv(K, alpha_vec, w0_scalar, y[:-1], delta)

training_data_prediction = k_func(X, X, d, alpha_vec, w0_scalar)
print(f"Training data predictions (wLS-SVM): \n{training_data_prediction}")

new_data = [(1, 1, None), (0.5, 1.5, None), (-0.5, 1.5, None), (-1, 1, None), (-0.5, 0.5, None), (0.5, 0.5, None)]

X_new = np.array(new_data)[:, :-1]
new_data_prediction = k_func(X_new, X, d, alpha_vec, w0_scalar)
print(f"New data predictions (wLS-SVM): \n{new_data_prediction}")

for i in range(n):
    print(f"Weighting for p_{i + 1} = {1 / V_inv[i][i]}")

########################################

finish_time = time()

print(f"\nThe code took: {finish_time - start_time} seconds to run\n")
