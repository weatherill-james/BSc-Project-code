import numpy as np
from time import time

start_time = time()

########################################

# Functions

# This is mathematically equivalent to the method described in the paper
k_func = lambda p_new, data, n, deg, alpha: sum(((1 + np.dot(p_new[:-1], data[p_index][:-1])) ** deg) * alpha[p_index] for p_index in range(n))

########################################

# We save our data in a: (x, y) format
data = [(-3, 9), (-2, 4), (-1, 1), (0, 0), (1, 1), (2, 4), (3, 9)]
# This converts our list of tuples to a NumPy array, which NumPy can then work with
data = np.array(data)

# Our number of training points is just the length of our data
n = len(data)

# Setting our parameters
d = 2
lamb = 0.01

# Explicitly computing the kernel matrix. This is mathematically equivalent to the method described in the paper, but can be done much more efficiently using matrix operations, as shown in the other files
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K[i, j] = (1 + np.dot(data[i][:-1], data[j][:-1])) ** d

# Our y vector is the last column of our data array
y = data[:, -1]

# Regularising our kernel matrix by adding lambda to the diagonal
K_regularized = K + lamb * np.eye(n)

# NumPy's linear solver uses LU decomposition in the background, and is much more efficient than doing LU decomposition ourselves
alpha_vec = np.linalg.solve(K_regularized, y)

new_data = [(-6, 36), (-4, 16), (-2.5, 6.25), (-1.5, 2.25), (-0.5, 0.25), (0.5, 0.25), (1.5, 2.25), (2.5, 6.25), (5, 25), (7, 49)]

for point in new_data:
    # We will use our kernel function to predict the values for our new data points
    prediction = k_func(point, data, n, d, alpha_vec)
    if len(point) > 2:
        print(f"\ny-hat for x = {point[:-1]}: {prediction}")
    else:
        print(f"\ny-hat for x = ({point[0]}): {prediction}")
    if point[-1] is not None:
        print(f" -> actual y: {point[-1]}")
    if point[0] != 0:
        print(f"% error: {abs(prediction - point[-1]) / abs(point[-1]) * 100}%")

########################################

finish_time = time()

print(f"\nThe code took: {finish_time - start_time} seconds to run\n")
