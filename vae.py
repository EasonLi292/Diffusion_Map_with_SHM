import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# 1. Dataset Class (The Data Pipeline)
class OscillatorDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Load images exactly as you did in the Diffusion Map script
        file_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.images = []
        for f in file_names:
            img = Image.open(os.path.join(image_dir, f)).convert('L') # Convert to Grayscale
            img = img.resize((100, 100)) # Ensure 100x100 as per resume
            self.images.append(np.array(img, dtype=np.float32) / 255.0) # Normalize 0-1
        
        self.images = np.array(self.images)
        self.images = torch.tensor(self.images).unsqueeze(1) # Shape: (N, 1, 100, 100)

    def __len__(self):
        # We return len - 1 because we need pairs (current, next)
        return len(self.images) - 1

    def __getitem__(self, idx):
        # Returns: (Current Frame, Next Frame)
        # This setup supports the "Predictive Modeling" claim
        return self.images[idx], self.images[idx+1]

# 2. The VAE Architecture (Feature Extraction)
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder: 100x100 -> Latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # 50x50
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 25x25
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 12x12
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent Space (Mean and LogVar)
        self.fc_mu = nn.Linear(128 * 12 * 12, latent_dim)
        self.fc_logvar = nn.Linear(128 * 12 * 12, latent_dim)
        
        # Decoder: Latent -> 100x100
        self.decoder_input = nn.Linear(latent_dim, 128 * 12 * 12)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),  # 50x50 (padding=0 adjusts size)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 100x100
            nn.Sigmoid() # Output pixels 0-1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        d = self.decoder_input(z)
        d = d.view(-1, 128, 12, 12)
        recon = self.decoder(d)
        return recon, mu, logvar, z

# 3. The Latent Predictor (The "Physics" Learner) 
# This small model learns to predict z(t+1) given z(t)
class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=2):
        super(LatentDynamics, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
    
    def forward(self, z):
        return self.net(z)

# 4. Training & Testing Functions

def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _kl_divergence(mu, logvar):
    # Compute KL per-sample then average to keep loss scale stable
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

def _batch_ssim(predicted_frame, target_next_frame):
    scores = []
    for i in range(predicted_frame.size(0)):
        img_true = target_next_frame[i].squeeze().cpu().numpy()
        img_pred = predicted_frame[i].squeeze().cpu().numpy()
        min_side = min(img_true.shape[-2], img_true.shape[-1])
        win_size = 7 if min_side >= 7 else max(3, min_side | 1)  # ensure odd and at least 3
        scores.append(ssim(img_true, img_pred, data_range=1.0, win_size=win_size))
    return float(np.mean(scores))

def train_system(image_dir, epochs=50, batch_size=32, val_split=0.2, device=None):
    # Setup
    dataset = OscillatorDataset(image_dir)
    if device is None:
        device = _select_device()

    if val_split > 0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * val_split))
        val_size = min(val_size, len(dataset) - 1)
    else:
        val_size = 0
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_ds = dataset
        val_loader = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    vae = VAE(latent_dim=2).to(device)
    dynamics = LatentDynamics(latent_dim=2).to(device)
    optimizer = optim.Adam(list(vae.parameters()) + list(dynamics.parameters()), lr=1e-3)

    print(f"Starting training on {train_size} train pairs" + (f" with {val_size} val pairs." if val_size else "."))
    history = []

    for epoch in range(epochs):
        vae.train()
        dynamics.train()
        total_loss = 0.0
        for current_frame, next_frame in train_loader:
            current_frame = current_frame.to(device)
            next_frame = next_frame.to(device)
            optimizer.zero_grad()
            
            # 1. VAE Pass (Reconstruct current frame)
            recon_x, mu, logvar, z_t = vae(current_frame)
            
            # 2. Dynamics Pass (Predict next latent state)
            z_t1_pred = dynamics(z_t.detach()) # Detach because we don't want to update VAE based on dynamics yet
            
            # 3. Decode Predicted Future (Predict next frame from predicted latent)
            d_pred = vae.decoder_input(z_t1_pred)
            d_pred = d_pred.view(-1, 128, 12, 12)
            pred_next_frame = vae.decoder(d_pred)
            
            # Loss: Reconstruction + KL + Prediction Accuracy
            recon_loss = nn.functional.mse_loss(recon_x, current_frame, reduction='mean')
            pred_loss = nn.functional.mse_loss(pred_next_frame, next_frame, reduction='mean')
            kl_loss = _kl_divergence(mu, logvar)
            loss = recon_loss + kl_loss + pred_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * current_frame.size(0)
            
        avg_train_loss = total_loss / len(train_loader.dataset)
        record = {"epoch": epoch, "train_loss": avg_train_loss}
        if val_loader is not None:
            vae.eval()
            dynamics.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_ssim = 0.0
                val_count = 0
                for cur, nxt in val_loader:
                    cur = cur.to(device)
                    nxt = nxt.to(device)
                    recon_x, mu, logvar, z_t = vae(cur)
                    z_t1_pred = dynamics(z_t)
                    d_pred = vae.decoder_input(z_t1_pred)
                    d_pred = d_pred.view(-1, 128, 12, 12)
                    pred_next_frame = vae.decoder(d_pred)
                    recon_l = nn.functional.mse_loss(recon_x, cur, reduction='mean')
                    pred_l = nn.functional.mse_loss(pred_next_frame, nxt, reduction='mean')
                    kl_l = _kl_divergence(mu, logvar)
                    val_loss += (recon_l + pred_l + kl_l).item() * cur.size(0)
                    val_ssim += _batch_ssim(pred_next_frame, nxt) * cur.size(0)
                    val_count += cur.size(0)
                avg_val_loss = val_loss / len(val_loader.dataset)
                avg_val_ssim = val_ssim / val_count if val_count else 0.0
            record["val_loss"] = avg_val_loss
            record["val_ssim"] = avg_val_ssim
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train {avg_train_loss:.4f} | val {avg_val_loss:.4f} | val SSIM {avg_val_ssim*100:.2f}%")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train {avg_train_loss:.4f}")
        history.append(record)
            
    vae.training_history = history
    return vae, dynamics, val_loader

#   accuracy validation
def validate_accuracy(vae, dynamics, image_dir=None, loader=None, device=None):
    if loader is None:
        if image_dir is None:
            raise ValueError("Provide either a DataLoader or an image_dir for validation.")
        dataset = OscillatorDataset(image_dir)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    if device is None:
        device = next(vae.parameters()).device

    ssim_scores = []
    
    print("\n--- Running Accuracy Validation ---")
    vae.eval()
    dynamics.eval()
    
    with torch.no_grad():
        for current_frame, target_next_frame in loader:
            current_frame = current_frame.to(device)
            target_next_frame = target_next_frame.to(device)
            # Get Latent of current
            _, _, _, z_t = vae(current_frame)
            
            # Predict Next Latent
            z_t1_pred = dynamics(z_t)
            
            # Decode Predicted Future
            d_pred = vae.decoder_input(z_t1_pred)
            d_pred = d_pred.view(-1, 128, 12, 12)
            predicted_frame = vae.decoder(d_pred)
            
            # Calculate SSIM per-sample to avoid win_size errors on small batches
            ssim_scores.append(_batch_ssim(predicted_frame, target_next_frame))
    
    avg_accuracy = np.mean(ssim_scores) * 100
    print(f"Model Prediction Fidelity (SSIM): {avg_accuracy:.2f}%")

# --- Usage ---
# image_dir = '/Users/eason/Desktop/Oscillator-MachineLearning/harmonic_motion_images_new'
# vae_model, dyn_model = train_system(image_dir, epochs=100)
# validate_accuracy(vae_model, dyn_model, image_dir)

import matplotlib.pyplot as plt

def visualize_phase_space(vae, dynamics, loader, device=None):
    if device is None:
        device = next(vae.parameters()).device
        
    vae.eval()
    dynamics.eval()
    
    # Store latent coordinates
    z_history = []
    
    print("Extracting latent dynamics...")
    with torch.no_grad():
        for current_frame, _ in loader:
            current_frame = current_frame.to(device)
            
            # Get the latent mean (mu) - this is the "cleanest" coordinate
            # We skip 'z' (which has random noise) to see the pure trajectory
            _, mu, _, _ = vae(current_frame)
            
            # Store as numpy points
            z_history.append(mu.cpu().numpy())
            
    # Concatenate all batches
    z_points = np.concatenate(z_history, axis=0)
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))
    
    # 1. The Trajectory Plot (What the VAE "sees")
    plt.subplot(1, 2, 1)
    # We use 'c' (color) to show time progression if the data is sequential
    # If data is shuffled, this just shows density
    plt.scatter(z_points[:, 0], z_points[:, 1], c=range(len(z_points)), cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(label='Time / Frame Index')
    plt.title("Learned Latent Phase Space (z1 vs z2)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, alpha=0.3)

    # 2. The Vector Field (What the Dynamics Network "predicts")
    # This shows the "flow" of physics the model learned
    plt.subplot(1, 2, 2)
    
    # Create a grid of points covering the range of our data
    x_min, x_max = z_points[:, 0].min(), z_points[:, 0].max()
    y_min, y_max = z_points[:, 1].min(), z_points[:, 1].max()
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20)
    )
    
    # Convert grid to tensor for the model
    grid_tensor = torch.tensor(np.column_stack((grid_x.ravel(), grid_y.ravel())), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Predict where these points move next
        pred_next = dynamics(grid_tensor)
        
    # Calculate velocity vectors (Next - Current)
    flow = (pred_next - grid_tensor).cpu().numpy()
    u = flow[:, 0].reshape(grid_x.shape)
    v = flow[:, 1].reshape(grid_y.shape)
    
    plt.streamplot(grid_x, grid_y, u, v, color='orange', density=1.5)
    plt.scatter(z_points[:, 0], z_points[:, 1], s=1, color='blue', alpha=0.1) # Overlay real data faintly
    plt.title("Predicted Vector Field (The 'Physics' Engine)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    
    plt.tight_layout()
    plt.show()

# Run it
# visualize_phase_space(vae_model, dyn_model, val_loader)
# Note: Ensure val_loader is not shuffled (shuffle=False) if you want the color gradient to represent time correctly!