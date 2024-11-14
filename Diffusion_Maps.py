import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import center_of_mass
from sklearn.preprocessing import StandardScaler

# Load Images and Extract Vertical Positions
image_dir = '/Users/eason/Desktop/Oscillator-MachineLearning/harmonic_motion_images'
image_files = sorted([f for f in os.listdir(image_dir)])

positions = []

for i in range(len(image_files)):
    # Load the image
    img = Image.open(os.path.join(image_dir, image_files[i]))
    #ensure only 2 dimensions
    img_array = np.array(img).squeeze()
    '''
    optimization for potential noise factors in training
    '''
    # Threshold the image to create a binary image
    #threshold = 128  # Adjust threshold as needed
    #binary_img = img_array > threshold
    
    # Compute the center of mass
    com = center_of_mass(img_array)
    # output is (row, column)
    # Use the vertical position in this case, could be optimized
    positions.append(com[0])  

positions = np.array(positions)

#Construct Data Points Using Time-Delay Embedding
#could consider using more than 2 images for each data point
data = []

for i in range(len(positions) - 1):
    data_point = positions[i:i+2]
    data.append(data_point)

data = np.array(data)
#print(f'Data shape: {data.shape}')

# Compute Velocities to Match Data Length
# Average velocity over the embedding window
velocities = []
for i in range(len(positions) - 1):
    v = (positions[i + 1] - positions[i]) 
    velocities.append(v)
velocities = np.array(velocities)

# Concatenate Data and Velocities
data = np.column_stack((data, velocities))

# Normalize the Data
#kinda important
data_scaled = StandardScaler().fit_transform(data)

# Implement Diffusion Maps
distance_matrix = squareform(pdist(data_scaled, metric='euclidean'))

def gaussian_kernel(distance_matrix, epsilon):
    return np.exp(-distance_matrix ** 2 / epsilon)

epsilon = np.median(distance_matrix)
affinity_matrix = gaussian_kernel(distance_matrix, epsilon)

# Row-normalize the affinity matrix
row_sums = affinity_matrix.sum(axis=1)
P = affinity_matrix / row_sums[:, np.newaxis]

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(P)

# Take the real parts
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 5: Visualize the Embedding
non_trivial_eigenvectors = eigenvectors[:, 1:3]

plt.figure(figsize=(8, 6))
plt.scatter(non_trivial_eigenvectors[:, 0], non_trivial_eigenvectors[:, 1],
            c=np.arange(len(non_trivial_eigenvectors)), cmap='viridis')
plt.title('Diffusion Map Embedding')
plt.xlabel('First Non-Trivial Eigenvector')
plt.ylabel('Second Non-Trivial Eigenvector')
plt.colorbar(label='Data Point Index')
plt.grid(True)
plt.show()
