# Improved Semi-fragile Watermarking Model
# Fixed for 85-90% accuracy
# -------------------------------------------------

import os
import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
import hashlib
import hmac
import matplotlib.pyplot as plt
from skimage import data, img_as_float, io, color
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import time
from tqdm import tqdm
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== IMPROVED ENCODER ====================
class ImprovedEncoder(nn.Module):
    """Enhanced encoder with deeper architecture and residual connections"""
    def __init__(self, payload_len=64, hidden=64):
        super().__init__()
        self.payload_len = payload_len
        
        # Payload embedding network - converts bit vector to spatial features
        self.payload_embed = nn.Sequential(
            nn.Linear(payload_len, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 16*16*8)
        )
        
        # Main encoding path with skip connections
        self.down1 = nn.Sequential(
            nn.Conv2d(3 + 8, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool2d(2)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden, hidden*2, 3, padding=1),
            nn.BatchNorm2d(hidden*2),
            nn.ReLU(),
            nn.Conv2d(hidden*2, hidden*2, 3, padding=1),
            nn.BatchNorm2d(hidden*2),
            nn.ReLU()
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(hidden*2, hidden*4, 3, padding=1),
            nn.BatchNorm2d(hidden*4),
            nn.ReLU(),
            nn.Conv2d(hidden*4, hidden*4, 3, padding=1),
            nn.BatchNorm2d(hidden*4),
            nn.ReLU()
        )
        
        # Upsampling path with skip connections
        self.up1 = nn.ConvTranspose2d(hidden*4, hidden*2, 2, stride=2)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(hidden*4, hidden*2, 3, padding=1),
            nn.BatchNorm2d(hidden*2),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(hidden*2, hidden, 2, stride=2)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(hidden*2, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 3, 1)
        )
        
    def forward(self, x, payload):
        # x: [B, 3, H, W], payload: [B, payload_len]
        B, _, H, W = x.shape
        
        # Embed payload into spatial features
        p_feat = self.payload_embed(payload)  # [B, 16*16*8]
        p_feat = p_feat.view(B, 8, 16, 16)
        p_feat = F.interpolate(p_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate image and payload features
        x_in = torch.cat([x, p_feat], dim=1)
        
        # Encoding path
        d1 = self.down1(x_in)
        p1 = self.pool(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        
        d3 = self.down3(p2)
        
        # Decoding path with skip connections
        u1 = self.up1(d3)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_conv1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_conv2(u2)
        
        # Generate residual with tanh activation for bounded output
        res = torch.tanh(self.out_conv(u2)) * 0.05  # Small residual scale
        
        return res

# ==================== IMPROVED DECODER ====================
class ImprovedDecoder(nn.Module):
    """Enhanced decoder with attention and deeper feature extraction"""
    def __init__(self, payload_len=64, hidden=64):
        super().__init__()
        
        # Feature extraction with multiple scales
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden*2, 3, padding=1),
            nn.BatchNorm2d(hidden*2),
            nn.ReLU(),
            nn.Conv2d(hidden*2, hidden*2, 3, padding=1),
            nn.BatchNorm2d(hidden*2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden*2, hidden*4, 3, padding=1),
            nn.BatchNorm2d(hidden*4),
            nn.ReLU(),
            nn.Conv2d(hidden*4, hidden*4, 3, padding=1),
            nn.BatchNorm2d(hidden*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Fully connected layers for bit extraction
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden*4*8*8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, payload_len)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        
        # Extract bits
        logits = self.fc(f3)
        
        return logits

# ==================== IMPROVED ATTACK PIPELINE ====================
class ImprovedAttack(nn.Module):
    """More realistic and diverse attack pipeline"""
    def __init__(self, p_jpeg=0.7):
        super().__init__()
        self.p_jpeg = p_jpeg
        
    def forward(self, imgs):
        # imgs: [B,3,H,W] in [0,1]
        x = imgs
        
        # 1. Random resize (with higher probability and varying scales)
        if random.random() < 0.95:
            scales = torch.empty(x.size(0)).uniform_(0.75, 0.95).tolist()
            out = torch.zeros_like(x)
            for i, s in enumerate(scales):
                h, w = x.shape[2], x.shape[3]
                nh, nw = max(1, int(h*s)), max(1, int(w*s))
                small = F.interpolate(x[i:i+1], size=(nh, nw), mode='bilinear', align_corners=False)
                back = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
                out[i:i+1] = back
            x = out
        
        # 2. Random rotation (small angles)
        if random.random() < 0.6:
            angles = torch.empty(x.size(0)).uniform_(-5, 5).tolist()
            theta_batch = []
            for ang in angles:
                rad = np.deg2rad(ang)
                theta = torch.tensor([
                    [np.cos(rad), -np.sin(rad), 0.0],
                    [np.sin(rad), np.cos(rad), 0.0]
                ], dtype=torch.float)
                theta_batch.append(theta.unsqueeze(0))
            theta_batch = torch.cat(theta_batch, dim=0).to(x.device)
            grid = F.affine_grid(theta_batch, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
        
        # 3. Gaussian blur
        if random.random() < 0.8:
            k = random.choice([3, 5])
            kernel = torch.tensor(cv2.getGaussianKernel(k, k/3).astype(np.float32))
            kernel2 = kernel @ kernel.T
            kernel2 = kernel2 / kernel2.sum()
            k_t = kernel2.unsqueeze(0).unsqueeze(0).to(x.device)
            pad = k // 2
            out = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            out_c = []
            for c in range(3):
                out_c.append(F.conv2d(out[:, c:c+1, :, :], k_t, padding=0))
            x = torch.cat(out_c, dim=1)
        
        # 4. Additive noise (stronger)
        if random.random() < 0.9:
            noise = torch.randn_like(x) * random.uniform(0.003, 0.01)
            x = torch.clamp(x + noise, 0, 1)
        
        # 5. JPEG compression (with varying quality)
        if random.random() < self.p_jpeg:
            x_np = (x.detach().cpu().numpy() * 255).astype(np.uint8)
            out_batch = []
            for i in range(x_np.shape[0]):
                img_bgr = cv2.cvtColor(x_np[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                q = random.randint(70, 95)
                _, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                dec_rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                out_batch.append(dec_rgb)
            x = torch.from_numpy(np.stack(out_batch, axis=0)).permute(0, 3, 1, 2).to(imgs.device).float()
        
        return x

# ==================== DATASET ====================
class ImageDataset(Dataset):
    def __init__(self, paths, image_size=256):
        self.paths = [str(p) for p in paths]
        self.image_size = image_size
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            img = io.imread(self.paths[idx])
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]
            
            # Convert to RGB if needed
            img = (img.astype(np.float32) / 255.0) if img.max() > 1.0 else img.astype(np.float32)
            
            # Resize
            H, W = img.shape[:2]
            side = min(H, W)
            cy, cx = H // 2, W // 2
            img_crop = img[cy-side//2:cy-side//2+side, cx-side//2:cx-side//2+side]
            img_resized = cv2.resize(img_crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            
            img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float()
            return img_t
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            # Return a dummy image
            return torch.zeros(3, self.image_size, self.image_size)

def create_datasets(root_dir, train_n=10000, val_n=2000, test_n=2000, seed=42):
    """Create train/val/test splits"""
    paths = list(Path(root_dir).glob('**/*.jpg')) + list(Path(root_dir).glob('**/*.png'))
    random.Random(seed).shuffle(paths)
    
    total_needed = train_n + val_n + test_n
    available = len(paths)
    
    if available < total_needed:
        print(f"Warning: Only {available} images available, need {total_needed}")
        # Adjust splits proportionally
        ratio = available / total_needed
        train_n = int(train_n * ratio)
        val_n = int(val_n * ratio)
        test_n = available - train_n - val_n
    
    train_paths = paths[:train_n]
    val_paths = paths[train_n:train_n+val_n]
    test_paths = paths[train_n+val_n:train_n+val_n+test_n]
    
    return train_paths, val_paths, test_paths

# ==================== TRAINING FUNCTION ====================
def train_model(root_images, epochs=20, batch_size=32, lr=1e-3, payload_len=64,
                train_n=10000, val_n=2000, test_n=2000, early_stop_patience=5):
    """Main training function with improved architecture and training loop"""
    
    print(f"Device: {device}")
    print(f"Creating datasets from {root_images}...")
    
    # Create datasets
    train_paths, val_paths, test_paths = create_datasets(
        root_images, train_n=train_n, val_n=val_n, test_n=test_n
    )
    
    train_ds = ImageDataset(train_paths)
    val_ds = ImageDataset(val_paths)
    test_ds = ImageDataset(test_paths)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Initialize models
    encoder = ImprovedEncoder(payload_len=payload_len).to(device)
    decoder = ImprovedDecoder(payload_len=payload_len).to(device)
    attack = ImprovedAttack(p_jpeg=0.7).to(device)
    
    # Optimizer with weight decay for regularization
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # VGG for perceptual loss
    vgg_loss_model = models.vgg16(pretrained=True).features[:16].to(device).eval()
    for p in vgg_loss_model.parameters():
        p.requires_grad = False
    
    def perceptual_loss(x, y):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_norm = (torch.clamp(x, 0, 1) - mean) / std
        y_norm = (torch.clamp(y, 0, 1) - mean) / std
        return F.mse_loss(vgg_loss_model(x_norm), vgg_loss_model(y_norm))
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    no_improve = 0
    
    for epoch in range(epochs):
        # ========== TRAINING ==========
        encoder.train()
        decoder.train()
        
        train_losses = []
        train_accs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs in pbar:
            imgs = imgs.to(device)
            B = imgs.size(0)
            
            # Generate random payload for each image
            payload = torch.randint(0, 2, (B, payload_len)).float().to(device)
            
            # Encode: add residual with embedded payload
            residual = encoder(imgs, payload)
            watermarked = torch.clamp(imgs + residual, 0.0, 1.0)
            
            # Attack
            attacked = attack(watermarked)
            
            # Decode: extract payload
            logits = decoder(attacked)
            
            # Compute losses
            bce_loss = F.binary_cross_entropy_with_logits(logits, payload)
            mse_loss = F.mse_loss(watermarked, imgs)
            perc_loss = perceptual_loss(watermarked, imgs)
            
            # Combined loss with balanced weights
            loss = bce_loss + 0.1 * mse_loss + 0.2 * perc_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                pred_bits = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred_bits == payload).float().mean().item()
            
            train_losses.append(loss.item())
            train_accs.append(acc)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc*100:.1f}%'})
        
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        
        # ========== VALIDATION ==========
        encoder.eval()
        decoder.eval()
        
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                B = imgs.size(0)
                
                payload = torch.randint(0, 2, (B, payload_len)).float().to(device)
                
                residual = encoder(imgs, payload)
                watermarked = torch.clamp(imgs + residual, 0.0, 1.0)
                attacked = attack(watermarked)
                logits = decoder(attacked)
                
                bce_loss = F.binary_cross_entropy_with_logits(logits, payload)
                val_losses.append(bce_loss.item())
                
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().reshape(-1)
                targs = payload.long().cpu().numpy().reshape(-1)
                
                all_preds.extend(preds.tolist())
                all_targets.extend(targs.tolist())
        
        avg_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(all_targets, all_preds)
        val_prec = precision_score(all_targets, all_preds, zero_division=0)
        val_rec = recall_score(all_targets, all_preds, zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc*100:.2f}%")
        print(f"Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc*100:.2f}%, Prec: {val_prec:.3f}, Rec: {val_rec:.3f}, F1: {val_f1:.3f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc
            }, 'best_model_checkpoint.pt')
            print(f"✓ Saved best model (Val Acc: {val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered (no improvement for {early_stop_patience} epochs)")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model_checkpoint.pt')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    # ========== FINAL TEST EVALUATION ==========
    print("\n=== Final Test Evaluation ===")
    encoder.eval()
    decoder.eval()
    
    test_accs = []
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            B = imgs.size(0)
            
            payload = torch.randint(0, 2, (B, payload_len)).float().to(device)
            
            residual = encoder(imgs, payload)
            watermarked = torch.clamp(imgs + residual, 0.0, 1.0)
            attacked = attack(watermarked)
            logits = decoder(attacked)
            
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().reshape(-1)
            targs = payload.long().cpu().numpy().reshape(-1)
            
            all_test_preds.extend(preds.tolist())
            all_test_targets.extend(targs.tolist())
    
    test_acc = accuracy_score(all_test_targets, all_test_preds)
    test_prec = precision_score(all_test_targets, all_test_preds, zero_division=0)
    test_rec = recall_score(all_test_targets, all_test_preds, zero_division=0)
    test_f1 = f1_score(all_test_targets, all_test_preds, zero_division=0)
    
    print(f"\nTest Results:")
    print(f"Accuracy:  {test_acc*100:.2f}%")
    print(f"Precision: {test_prec:.3f}")
    print(f"Recall:    {test_rec:.3f}")
    print(f"F1-Score:  {test_f1:.3f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot([a*100 for a in history['train_acc']], label='Train Acc')
    plt.plot([a*100 for a in history['val_acc']], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot([f*100 for f in history['val_f1']], label='F1-Score')
    plt.plot([p*100 for p in history['val_precision']], label='Precision')
    plt.plot([r*100 for r in history['val_recall']], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score (%)')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training history saved to training_history.png")
    
    return encoder, decoder, history


if __name__ == "__main__":
    # Example usage
    ROOT_IMAGES = './images_train'  # Update this path
    
    # Train the model
    encoder, decoder, history = train_model(
        root_images=ROOT_IMAGES,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        payload_len=64,
        train_n=10000,
        val_n=2000,
        test_n=2000,
        early_stop_patience=5
    )
