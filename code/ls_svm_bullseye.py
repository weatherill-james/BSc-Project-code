import numpy as np
from time import time

start_time = time()

########################################

# Functions

k_func = lambda p_new, data, n, deg, alpha: sum(((1 + np.dot(p_new[:-1], data[p_index][:-1])) ** deg) * alpha[p_index] for p_index in range(n))

# We set up a prediction function to make it clearer how we are using the kernel
# Other code files disregard this function, since there are more computationally efficient ways to achieve this same result
predict_func = lambda p_new, data, n, deg, alpha, w_0: k_func(p_new, data, n, deg, alpha) + w_0

########################################

# We save our data in a: (x_1, x_2, y) format
data = [(0, 0, 1), (0.5, 0, 1), (0.35, 0.35, 1), (0, 0.5, 1), (-0.35, 0.35, 1), (-0.5, 0, 1), (-0.35, -0.35, 1), (0, -0.5, 1), (0.35, -0.35, 1), (1, 0, -1), (0.87, 0.5, -1), (0.5, 0.87, -1), (0, 1, -1), (-0.5, 0.87, -1), (-0.87, 0.5, -1), (-1, 0, -1), (-0.87, -0.5, -1), (-0.5, -0.87, -1), (0, -1, -1), (0.5, -0.87, -1), (0.87, -0.5, -1)]
data = np.array(data)

n = len(data)

d = 2
lamb = 0.01

# The upgraded way to computed the kernel matrix, which gives identical results to the explicit method, but is much more efficient
X = data[:, :-1]
K = (1 + X @ X.T) ** d

# This is our RHS vector, which is the y vector with a '0' as the last element beneath it
y = np.append(data[:, -1], 0)

# The ones vector that will border our kernel matrix
ones_vec = np.ones((n, 1))

K_regularized = K + lamb * np.eye(n)

LHS_matrix = np.block([[K_regularized, ones_vec],
                       [ones_vec.T,        0   ]])

# We split our solution into the alpha vector, and the w0 value
solution = np.linalg.solve(LHS_matrix, y)
alpha_vec = solution[:-1]
w0_scalar = solution[-1]

# training_data_prediction = np.array([predict_func(p, data, n, degree, alpha_vec, w0_scalar) for p in data])
# print(training_data_prediction)

new_data = [(0.5, 0.5, None), (-1, 1, None), (-0.75, 0.5, None), (-0.71, -0.71, None), (0, -0.6, None), (0.35, -0.25, None)]

new_data_prediction = np.array([predict_func(p, data, n, d, alpha_vec, w0_scalar) for p in new_data])
print(new_data_prediction)

########################################

finish_time = time()

print(f"\nThe code took: {finish_time - start_time} seconds to run\n")
