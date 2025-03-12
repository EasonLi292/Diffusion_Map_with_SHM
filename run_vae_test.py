import sys
sys.path.insert(0, '/Users/eason/Desktop/Oscillator-MachineLearning/Diffusion_Map_with_SHM')
from vae import train_system, validate_accuracy

# Run training
image_dir = '/Users/eason/Desktop/Oscillator-MachineLearning/harmonic_motion_images_new'
print("Starting VAE training...")
vae_model, dyn_model, val_loader = train_system(image_dir, epochs=15, batch_size=16)

# Validate accuracy
print("\nValidating model accuracy...")
validate_accuracy(vae_model, dyn_model, loader=val_loader)

print("\nTraining complete!")
