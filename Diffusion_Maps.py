import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# Load Images and Flatten
image_dir = '/Users/eason/Desktop/Oscillator-MachineLearning/harmonic_motion_images_new'
image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])

images = []
for file in image_files:
    img = Image.open(os.path.join(image_dir, file))
    img_array = np.array(img).flatten()
    images.append(img_array)

images = np.array(images)

# Compute Velocities (Differences Between Consecutive Images)
velocities = (images[1:] - images[:-1])
images = images[:-1]  # Adjust images to match velocities

# Normalize images and velocities separately
images_scaled = StandardScaler().fit_transform(images) 
velocities_scaled = StandardScaler().fit_transform(velocities) * 1.4
data_scaled = np.hstack((images_scaled, velocities_scaled))


# Compute pairwise distances between data points
distance_matrix = squareform(pdist(data_scaled))

# Define the Gaussian Kernel
def gauss_kernel(distance_matrix, epsilon):
    return np.exp(-distance_matrix**2 / (2 * epsilon**2))
epsilon = np.median(distance_matrix) 

# Compute the affinity matrix
affinity_matrix = gauss_kernel(distance_matrix, epsilon)
q = affinity_matrix.sum(axis=1)

# Symmetric normalization: K(i,j) = W(i,j) / sqrt(q(i)*q(j))
K = affinity_matrix / np.sqrt(np.outer(q, q))

# Row-normalize K to get Markov matrix P
row_sums_K = K.sum(axis=1)
# Row-normalize the affinity matrix
row_sums = affinity_matrix.sum(axis=1)
P = K / row_sums_K[:, np.newaxis]
P = np.linalg.matrix_power(P, 5)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(P)
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Visualize the Diffusion Map
non_trivial_eigenvectors = eigenvectors[:, 1:3]

plt.figure(figsize=(8, 6))
plt.scatter(non_trivial_eigenvectors[:, 0], non_trivial_eigenvectors[:, 1],
            c=np.arange(len(non_trivial_eigenvectors)), cmap='viridis')
plt.title('Diffusion Map Embedding with Gaussian Kernel')
plt.xlabel('First Non-Trivial Eigenvector')
plt.ylabel('Second Non-Trivial Eigenvector')
plt.colorbar(label='Data Point Index')
plt.grid(True)
plt.show()