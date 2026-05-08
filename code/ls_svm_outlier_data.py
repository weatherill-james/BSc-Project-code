import numpy as np
from time import time

start_time = time()

########################################

# Functions

k_func = lambda p_new, data, n, deg, alpha: sum(((1 + np.dot(p_new[:-1], data[p_index][:-1])) ** deg) * alpha[p_index] for p_index in range(n))

predict_func = lambda p_new, data, n, deg, alpha, w_0: k_func(p_new, data, n, deg, alpha) + w_0

########################################

# (x_1, x_2, y)
data = [(-2.5, 1.5, 1), (-3.5, 1.5, 1), (-3.5, 0.5, 1), (-2.5, 0.5, 1), (-3, 1, 1), (-6, 2, 1), (-5, 1, 1), (-4, 0, 1), (-4, 2, 1), (-6, 0, 1), (5.5, 1.5, 1), (5, 0.5, 1), (6, 1, 1), (2.5, 1.5, -1), (3.5, 1.5, -1), (3.5, 0.5, -1), (2.5, 0.5, -1), (3, 1, -1), (6, 0, -1), (5, 1, -1), (4, 0, -1), (4, 2, -1), (6, 2, -1)]
data = np.array(data)

n = len(data)

d = 2
lamb = 0.01

X = data[:, :-1]
K = (1 + X @ X.T) ** d

y = np.append(data[:, -1], 0)

ones_vec = np.ones((n, 1))

K_regularized = K + lamb * np.eye(n)

# This is our bordered kernel matrix, exactly as described in the paper
LHS_matrix = np.block([[K_regularized, ones_vec],
                       [ones_vec.T,        0   ]])

solution = np.linalg.solve(LHS_matrix, y)
alpha_vec = solution[:-1]
w0_scalar = solution[-1]

training_data_prediction = np.array([predict_func(p, data, n, d, alpha_vec, w0_scalar) for p in data])
print(f"Training data predictions (LS-SVM): \n{training_data_prediction}")

new_data = [(1, 1, None), (0.5, 1.5, None), (-0.5, 1.5, None), (-1, 1, None), (-0.5, 0.5, None), (0.5, 0.5, None)]

new_data_prediction = np.array([predict_func(p, data, n, d, alpha_vec, w0_scalar) for p in new_data])
print(f"New data predictions (LS-SVM): \n{new_data_prediction}")

########################################

finish_time = time()

print(f"\nThe code took: {finish_time - start_time} seconds to run\n")
