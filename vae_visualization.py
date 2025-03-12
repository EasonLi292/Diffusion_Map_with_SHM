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

# Create sample predictions visualization
print("\nGenerating sample predictions...")
dataset = OscillatorDataset(image_dir)
device = next(vae_model.parameters()).device

vae_model.eval()
dyn_model.eval()

fig, axes = plt.subplots(3, 6, figsize=(15, 8))
with torch.no_grad():
    for i in range(3):
        idx = i * 20  # Sample every 20 frames
        current_frame, target_next = dataset[idx]
        current_frame = current_frame.unsqueeze(0).to(device)
        target_next = target_next.unsqueeze(0).to(device)

        # Get prediction
        _, _, _, z_t = vae_model(current_frame)
        z_t1_pred = dyn_model(z_t)
        d_pred = vae_model.decoder_input(z_t1_pred)
        d_pred = d_pred.view(-1, 128, 12, 12)
        pred_next = vae_model.decoder(d_pred)

        # Plot current frame
        axes[i, 0].imshow(current_frame.squeeze().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('Current Frame', fontsize=10)
        axes[i, 0].axis('off')

        # Plot target next frame
        axes[i, 1].imshow(target_next.squeeze().cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('Target Next', fontsize=10)
        axes[i, 1].axis('off')

        # Plot predicted next frame
        axes[i, 2].imshow(pred_next.squeeze().cpu().numpy(), cmap='gray')
        axes[i, 2].set_title('Predicted Next', fontsize=10)
        axes[i, 2].axis('off')

        # Plot difference
        diff = np.abs(target_next.squeeze().cpu().numpy() - pred_next.squeeze().cpu().numpy())
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title('Difference', fontsize=10)
        axes[i, 3].axis('off')

        # Reconstruct current frame
        recon_current, _, _, _ = vae_model(current_frame)
        axes[i, 4].imshow(recon_current.squeeze().cpu().numpy(), cmap='gray')
        axes[i, 4].set_title('Reconstructed Current', fontsize=10)
        axes[i, 4].axis('off')

        # Show latent space position
        axes[i, 5].scatter(z_t.cpu().numpy()[0, 0], z_t.cpu().numpy()[0, 1], s=100, c='blue')
        axes[i, 5].set_xlim(-3, 3)
        axes[i, 5].set_ylim(-3, 3)
        axes[i, 5].set_title('Latent Position', fontsize=10)
        axes[i, 5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_sample_predictions.png', dpi=300, bbox_inches='tight')
print("Saved vae_sample_predictions.png")

print("\nVisualization complete!")
