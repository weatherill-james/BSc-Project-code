import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

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

# This function grabs a random sample of 5 images for each class in 'classes', and returns them as a list of 28x28 arrays, ready to be plotted
def get_samples(X, y, classes, n = 5):
    images = []
    labels = []

    for c in classes:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size = n, replace = False)

        for i in chosen:
            images.append(X[i].reshape(28, 28))
            labels.append(c)
    return images, labels

########################################

# Grabbing the MNIST data and labels

assets_path = os.path.join(os.getcwd(), "..", "assets")
latex_assets_path = os.path.join(os.getcwd(), "..", "..", "LaTeX", "Dissertation_JWeatherill", "inputs", "assets")

X_training = parse_data(os.path.join(assets_path, "MNIST_training_data_60k.gz"))
y_training = parse_labels(os.path.join(assets_path, "MNIST_training_labels_60k.gz"))

X_testing  = parse_data(os.path.join(assets_path, "MNIST_testing_data_10k.gz"))
y_testing  = parse_labels(os.path.join(assets_path, "MNIST_testing_labels_10k.gz"))

########################################

# Using the same seed as we use for the actual model
rng = np.random.default_rng(3)

digits_and_filenames = [([0, 1, 2, 3, 4], os.path.join(latex_assets_path, "MNIST_training_0to4.pdf")),
                        ([5, 6, 7, 8, 9], os.path.join(latex_assets_path, "MNIST_training_5to9.pdf"))]

# The ideal file type is PDF, since it is vector graphics, and will look good at any size
for digits, filename in digits_and_filenames:
    images, labels = get_samples(X_training, y_training, digits, n = 5)

    fig, axes = plt.subplots(5, 5, figsize = (5, 5))

    for ax, image, label in zip(axes.flat, images, labels):
        ax.imshow(image, cmap = "gray", interpolation = "nearest")
        ax.axis("off")
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
    plt.savefig(filename, dpi = 300, bbox_inches = "tight")
    plt.close()
