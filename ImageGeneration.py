import numpy as np
import os
from PIL import Image, ImageDraw

# Parameters
mass = 1.0          # Mass (kg)
k = 1.0             # Spring constant (N/m)
dt = 1/24           # Time step (s)
num_steps = 300     # Number of frames

# Initial conditions
x = np.zeros(num_steps)
v = np.zeros(num_steps)
a = np.zeros(num_steps)

x[0] = 0.5          # Initial position (m)
v[0] = 0.0          # Initial velocity (m/s)
a[0] = - (k / mass) * x[0]  # Initial acceleration (m/s^2)

# Velocity Verlet Integration
for i in range(num_steps - 1):
    # Update position
    x[i + 1] = x[i] + v[i] * dt + 0.5 * a[i] * dt**2
    # Compute new acceleration
    a_new = - (k / mass) * x[i + 1]
    # Update velocity
    v[i + 1] = v[i] + 0.5 * (a[i] + a_new) * dt
    # Update acceleration for next iteration
    a[i + 1] = a_new

# Create directory to save images
image_dir = 'harmonic_motion_images_24fps'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Image parameters
img_size = (100, 100)  # Width x Height in pixels
sphere_radius = 5      # Radius of the sphere in pixels

# Adjust y_min and y_max to account for the sphere's radius
y_min = sphere_radius
y_max = img_size[1] - sphere_radius  # img_size[1] is the height dimension

# Scaling factor to convert physical position to pixel position
def scale_position(x, x_min, x_max, y_min, y_max):
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min
    y = y_min + ((x - x_min) / x_range) * y_range
    return y_max - y  # Invert y-axis to match image coordinate system

# Determine physical position range with a small buffer
buffer = 0.1  # Adjust as needed
x_min = np.min(x) - buffer
x_max = np.max(x) + buffer

# Generate and save images
for i in range(num_steps):
    # Create a black background image
    img = Image.new('RGB', img_size, color='black')
    draw = ImageDraw.Draw(img)
    # Scale the physical position to image coordinates
    y_pos = scale_position(x[i], x_min, x_max, y_min, y_max)
    # Draw the white sphere
    left = img_size[0] // 2 - sphere_radius
    right = img_size[0] // 2 + sphere_radius
    top = int(y_pos) - sphere_radius
    bottom = int(y_pos) + sphere_radius
    draw.ellipse([left, top, right, bottom], fill='white')
    # Save the image
    filename = os.path.join(image_dir, f'frame_{i:03d}.png')
    img.save(filename)


import matplotlib.pyplot as plt

time = np.arange(num_steps) * dt

plt.figure(figsize=(8, 6))
plt.plot(time, x, label='Position (x)')
plt.title('Position vs. Time for Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()
plt.show()
