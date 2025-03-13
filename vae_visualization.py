import sys
sys.path.insert(0, '/Users/eason/Desktop/Oscillator-MachineLearning/Diffusion_Map_with_SHM')
from vae import train_system, validate_accuracy, OscillatorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

# Run training
image_dir = '/Users/eason/Desktop/Oscillator-MachineLearning/harmonic_motion_images_new'
print("Starting VAE training...")
vae_model, dyn_model, val_loader = train_system(image_dir, epochs=50, batch_size=16)

# Extract training history
history = vae_model.training_history
epochs = [h['epoch'] for h in history]
train_loss = [h['train_loss'] for h in history]
val_loss = [h['val_loss'] for h in history]
val_ssim = [h['val_ssim'] * 100 for h in history]

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss curves
ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2)
ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot SSIM accuracy
ax2.plot(epochs, val_ssim, label='Val SSIM', linewidth=2, color='green')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('SSIM (%)', fontsize=12)
ax2.set_title('Validation SSIM Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_training_metrics.png', dpi=300, bbox_inches='tight')
print("Saved vae_training_metrics.png")

# Validate accuracy
print("\nValidating model accuracy...")
validate_accuracy(vae_model, dyn_model, loader=val_loader)

# Create latent space visualization
print("\nGenerating latent space visualization...")
dataset = OscillatorDataset(image_dir)
device = next(vae_model.parameters()).device

vae_model.eval()
dyn_model.eval()

# Extract all latent representations
z_history = []
with torch.no_grad():
    for i in range(len(dataset)):
        current_frame, _ = dataset[i]
        current_frame = current_frame.unsqueeze(0).to(device)
        _, mu, _, _ = vae_model(current_frame)
        z_history.append(mu.cpu().numpy())

z_points = np.concatenate(z_history, axis=0)

# Create comprehensive latent space visualization
fig = plt.figure(figsize=(16, 6))

# 1. Latent trajectory with time coloring
ax1 = plt.subplot(1, 3, 1)
scatter = ax1.scatter(z_points[:, 0], z_points[:, 1], c=range(len(z_points)),
                      cmap='viridis', s=30, alpha=0.7)
plt.colorbar(scatter, label='Frame Index', ax=ax1)
ax1.set_title("Latent Space Trajectory", fontsize=14, fontweight='bold')
ax1.set_xlabel("Latent Dimension 1", fontsize=12)
ax1.set_ylabel("Latent Dimension 2", fontsize=12)
ax1.grid(True, alpha=0.3)

# 2. Vector field showing dynamics
ax2 = plt.subplot(1, 3, 2)
x_min, x_max = z_points[:, 0].min() - 0.5, z_points[:, 0].max() + 0.5
y_min, y_max = z_points[:, 1].min() - 0.5, z_points[:, 1].max() + 0.5

grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, 20),
    np.linspace(y_min, y_max, 20)
)

grid_tensor = torch.tensor(np.column_stack((grid_x.ravel(), grid_y.ravel())),
                          dtype=torch.float32).to(device)

with torch.no_grad():
    pred_next = dyn_model(grid_tensor)

flow = (pred_next - grid_tensor).cpu().numpy()
u = flow[:, 0].reshape(grid_x.shape)
v = flow[:, 1].reshape(grid_y.shape)

ax2.streamplot(grid_x, grid_y, u, v, color='orange', density=1.5, linewidth=1)
ax2.scatter(z_points[:, 0], z_points[:, 1], s=10, color='blue', alpha=0.3)
ax2.set_title("Learned Dynamics Vector Field", fontsize=14, fontweight='bold')
ax2.set_xlabel("Latent Dimension 1", fontsize=12)
ax2.set_ylabel("Latent Dimension 2", fontsize=12)

# 3. Latent space with arrows showing transitions
ax3 = plt.subplot(1, 3, 3)
ax3.scatter(z_points[:, 0], z_points[:, 1], c=range(len(z_points)),
           cmap='viridis', s=30, alpha=0.5)
# Draw arrows for every 5th transition
for i in range(0, len(z_points)-1, 5):
    ax3.arrow(z_points[i, 0], z_points[i, 1],
             z_points[i+1, 0] - z_points[i, 0],
             z_points[i+1, 1] - z_points[i, 1],
             head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.4, linewidth=0.5)
ax3.set_title("Frame-to-Frame Transitions", fontsize=14, fontweight='bold')
ax3.set_xlabel("Latent Dimension 1", fontsize=12)
ax3.set_ylabel("Latent Dimension 2", fontsize=12)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_latent_space.png', dpi=300, bbox_inches='tight')
print("Saved vae_latent_space.png")

print("\nVisualization complete!")
