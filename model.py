# Semi-fragile Watermarking Colab Notebook
# -------------------------------------------------
# This is a Colab-ready Python script (copy cells into a Colab notebook
# or save as .py and open in Colab). It implements:
# feature extraction (VGG mid-level) -> HMAC digest -> DWT+DCT+SVD embedding
# into N SIFT patches -> attack simulator -> extractor + decision logic -> evaluation plots
#
# Requirements (first cell): install packages

# %%
# Install dependencies (run this cell first in Colab)
!pip install -q torch torchvision matplotlib opencv-python-headless scikit-image scikit-learn pywt Pillow tqdm

# Note: opencv-python-headless includes SIFT in many builds. If your CV2 lacks SIFT,
# use `opencv-contrib-python` instead. In that case replace the install line above.

# %%
# Imports
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
import torch
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

# %%
# Utility helpers

def pil_to_cv2(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_cv2):
    if img_cv2.ndim == 2:
        from PIL import Image
        return Image.fromarray(img_cv2)
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))


def to_y_channel(rgb_img):
    # rgb_img in range [0,1] or [0,255]
    img = rgb_img.copy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = ycbcr[:, :, 0].astype(np.float32) / 255.0
    return Y

# %%
# Feature extractor (VGG conv3_3 pooled)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vgg = models.vgg16(pretrained=True).features.to(device).eval()
# We'll extract activation after layer 10 (conv3_3-ish)
layer_idx = 10

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

from torch import nn

class VGGHook():
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.activation = None
        self.handle = self.model[self.layer_idx].register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.activation = output.detach()
    def close(self):
        self.handle.remove()

hook = VGGHook(vgg, layer_idx)


def extract_vgg_descriptor(rgb_img, pca=None):
    # rgb_img: HxWx3 uint8 RGB
    img_t = transform(rgb_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = vgg(img_t)
        feat = hook.activation  # shape [1, C, H, W]
        pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1)).squeeze().cpu().numpy()
    if pca is not None:
        return pca.transform(pooled.reshape(1,-1)).ravel()
    return pooled.ravel()

# %%
# PCA helper (to reduce feature dimension)

def fit_pca_on_sample_images(sample_images, n_comp=128):
    feats = []
    for img in sample_images:
        f = extract_vgg_descriptor(img)
        feats.append(f)
    feats = np.stack(feats, axis=0)
    pca = PCA(n_components=n_comp)
    pca.fit(feats)
    return pca

# %%
# HMAC digest

def compute_hmac_digest(feature_vec, key=b'secret_key', digest_bits=128):
    # feature_vec: 1D float array
    # We quantize the feature vector to bytes deterministically
    feat_bytes = feature_vec.astype(np.float32).tobytes()
    mac = hmac.new(key, feat_bytes, hashlib.sha256).digest()
    # convert to bitstring and truncate
    bitstr = ''.join(f'{b:08b}' for b in mac)
    return bitstr[:digest_bits]

# %%
# DWT + DCT + SVD embedding helpers

def dwt2_channel(channel):
    coeffs = pywt.dwt2(channel, 'haar')
    LL, (LH, HL, HH) = coeffs
    return coeffs


def idwt2_channel(coeffs):
    return pywt.idwt2(coeffs, 'haar')


def block_dct(block):
    # block: float32
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def block_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def svd_modify_and_reconstruct(A, Wbits, alpha=0.01):
    # A: 2D array (DCT coefficients). We'll perturb singular values according to bits
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_mod = S.copy()
    # embed bits into the largest k singular values' LSB by small additive offset
    k = min(len(Wbits), len(S_mod))
    for i in range(k):
        bit = int(Wbits[i])
        # map bit 0->-1, 1->+1 small change relative to singular value
        S_mod[i] = S_mod[i] + alpha * (1 if bit==1 else -1) * (np.mean(S) + 1e-8)
    A_mod = U @ np.diag(S_mod) @ Vt
    return A_mod

# %%
# SIFT patching helpers

def get_sift_keypoints(rgb_img, max_kp=64):
    # expects uint8 RGB
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kps = sift.detect(gray, None)
    kps = sorted(kps, key=lambda x: -x.response)[:max_kp]
    return kps


def extract_patch(img, kp, patch_size=64):
    x,y = int(kp.pt[0]), int(kp.pt[1])
    half = patch_size // 2
    h,w = img.shape[:2]
    x1,x2 = max(0,x-half), min(w, x+half)
    y1,y2 = max(0,y-half), min(h, y+half)
    patch = img[y1:y2, x1:x2]
    # if patch smaller than desired, pad
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.copyMakeBorder(patch, 
                                   top=0, bottom=patch_size-patch.shape[0],
                                   left=0, right=patch_size-patch.shape[1],
                                   borderType=cv2.BORDER_REPLICATE)
    return patch, (x1,y1,x2,y2)

# %%
# Embedding routine across N SIFT patches

def embed_digest_in_image(rgb_img, digest_bits, N_patches=8, patch_size=64, alpha=0.02, key=b'secret'):
    # rgb_img: uint8 RGB
    kp_list = get_sift_keypoints(rgb_img, max_kp=N_patches*4)
    chosen = kp_list[:N_patches]
    Y = to_y_channel(rgb_img)  # float [0,1]
    Y_out = Y.copy()
    embed_locations = []
    bits_per_patch = len(digest_bits) // N_patches

    for i,kp in enumerate(chosen):
        patch, (x1,y1,x2,y2) = extract_patch(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), kp, patch_size)
        # work on Y patch
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        Yp = to_y_channel(patch_rgb)
        # DWT
        LL, (LH, HL, HH) = dwt2_channel(Yp)
        # We'll embed into LL's DCT by block-dct full patch
        dct_LL = block_dct(LL)
        start = i*bits_per_patch
        bits = digest_bits[start:start+bits_per_patch]
        # apply SVD modification on dct_LL
        dct_mod = svd_modify_and_reconstruct(dct_LL, bits, alpha=alpha)
        LL_mod = block_idct(dct_mod)
        # inverse DWT
        Yp_mod = idwt2_channel((LL_mod, (LH,HL,HH)))
        # place back Y_out
        hpatch,wpatch = Yp_mod.shape
        Y_out[y1:y1+hpatch, x1:x1+wpatch] = Yp_mod
        embed_locations.append((x1,y1,x1+wpatch,y1+hpatch))

    # reconstruct RGB from modified Y and original CbCr
    img = (rgb_img.copy()).astype(np.uint8)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycbcr[:,:,0] = np.clip(Y_out*255.0, 0, 255)
    out_rgb = cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return out_rgb, embed_locations

# %%
# Extraction routine from patches

def extract_digest_from_image(rgb_img, N_patches=8, patch_size=64, bits_per_patch=16, alpha=0.02):
    kp_list = get_sift_keypoints(rgb_img, max_kp=N_patches*4)
    chosen = kp_list[:N_patches]
    extracted_bits = []
    for i,kp in enumerate(chosen):
        patch, (x1,y1,x2,y2) = extract_patch(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), kp, patch_size)
        Yp = to_y_channel(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        LL, (LH,HL,HH) = dwt2_channel(Yp)
        dct_LL = block_dct(LL)
        # SVD and read relative sign of largest singular values compared to baseline
        U,S,Vt = np.linalg.svd(dct_LL, full_matrices=False)
        # derive bits from signs of (S - median(S)) as simple decoder
        med = np.median(S)
        patch_bits = ''.join('1' if (s-med)>0 else '0' for s in S[:bits_per_patch])
        extracted_bits.append(patch_bits)
    # concatenate (note: simple aggregation)
    return ''.join(extracted_bits)

# %%
# Attack simulator (benign + malicious)

def attack_jpeg(img, quality=75):
    # img: uint8 RGB
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    is_success, encimg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)


def attack_resize(img, scale=0.8):
    h,w = img.shape[:2]
    newh,neww = max(1,int(h*scale)), max(1,int(w*scale))
    small = cv2.resize(img, (neww,newh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)
    return back


def attack_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize,ksize), 0)


def attack_rotate(img, angle=10):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

# compound

def compound_benign(img):
    out = attack_resize(img, scale=0.9)
    out = attack_jpeg(out, quality=85)
    out = attack_blur(out, ksize=3)
    return out

# malicious example: splice patch from another image

def malicious_splice(img, donor):
    h,w = img.shape[:2]
    dh,dw = donor.shape[:2]
    # take central crop from donor and paste to center of img
    ch, cw = dh//2, dw//2
    patch = donor[ch-50:ch+50, cw-50:cw+50]
    out = img.copy()
    H,W = out.shape[:2]
    x1,y1 = W//2 - patch.shape[1]//2, H//2 - patch.shape[0]//2
    out[y1:y1+patch.shape[0], x1:x1+patch.shape[1]] = patch
    return out

# %%
# Decision logic

def hamming_distance(s1, s2):
    # equal-length bitstrings
    return sum(c1!=c2 for c1,c2 in zip(s1,s2))


def verify_semi_fragile(received_rgb, original_rgb, key=b'secret_key', N_patches=8, digest_bits=128, bits_per_patch=16, T_accept=12, T_reject=30):
    # recompute digest from received (for semi-fragile we want to recompute from received image features)
    # but using original features would be used for owner-side. Here we simulate both.
    # For user's verification: owner stores digest(original) and compares
    # For simplicity, owner digest computed from original
    pca = None
    orig_feat = extract_vgg_descriptor(original_rgb)
    orig_digest = compute_hmac_digest(orig_feat, key=key, digest_bits=digest_bits)

    # extract bits from received
    extracted = extract_digest_from_image(received_rgb, N_patches=N_patches, patch_size=64, bits_per_patch=bits_per_patch)
    # if extracted shorter than digest, pad/truncate
    extracted = extracted[:digest_bits]
    if len(extracted) < digest_bits:
        extracted = extracted.ljust(digest_bits, '0')

    # recompute digest from received features
    rec_feat = extract_vgg_descriptor(received_rgb)
    rec_digest = compute_hmac_digest(rec_feat, key=key, digest_bits=digest_bits)

    # compare
    ham1 = hamming_distance(extracted, orig_digest)
    ham2 = hamming_distance(extracted, rec_digest)
    # Decision: if extracted close to recomputed -> likely benign; if far from recomputed and far from original -> tampered
    if ham2 <= T_accept:
        return 'PASS', ham1, ham2, extracted, orig_digest, rec_digest
    elif ham2 > T_reject:
        return 'TAMPER', ham1, ham2, extracted, orig_digest, rec_digest
    else:
        return 'UNCERTAIN', ham1, ham2, extracted, orig_digest, rec_digest

# %%
# Quick demo run (example images)
from skimage import data
img = img_as_float(data.chelsea())  # cat image
img_uint8 = (img*255).astype(np.uint8)
# make a donor image
donor = img_as_float(data.astronaut())
donor_uint8 = (donor*255).astype(np.uint8)

# Fit PCA on small sample (optional) - omitted to keep runtime low

# compute feature and digest
feat = extract_vgg_descriptor(img_uint8)
digest = compute_hmac_digest(feat, key=b'my_secret_key', digest_bits=128)
print('Digest bits length:', len(digest))

# embed
watermarked, locs = embed_digest_in_image(img_uint8, digest, N_patches=8, patch_size=64, alpha=0.02)

# attack benign
attacked_benign = compound_benign(watermarked)
# attack malicious
attacked_malicious = malicious_splice(watermarked, donor_uint8)

# verify
res_benign = verify_semi_fragile(attacked_benign, img_uint8, key=b'my_secret_key')
res_mal = verify_semi_fragile(attacked_malicious, img_uint8, key=b'my_secret_key')
print('Benign result:', res_benign[0], 'hamming to orig,rec:', res_benign[1], res_benign[2])
print('Malicious result:', res_mal[0], 'hamming to orig,rec:', res_mal[1], res_mal[2])

# show images
plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title('Original'); plt.imshow(img_uint8); plt.axis('off')
plt.subplot(1,3,2); plt.title('Watermarked'); plt.imshow(watermarked); plt.axis('off')
plt.subplot(1,3,3); plt.title('Benign Attacked'); plt.imshow(attacked_benign); plt.axis('off')

# %%
# Evaluation harness (run more systematic tests)

def evaluate_on_dataset(images, donor_images, key=b'my_secret_key'):
    stats = {'benign_pass':0, 'benign_total':0, 'mal_pass':0, 'mal_total':0}
    for i,img in enumerate(images):
        img_uint8 = (img*255).astype(np.uint8)
        feat = extract_vgg_descriptor(img_uint8)
        digest = compute_hmac_digest(feat, key=key, digest_bits=128)
        watermarked, _ = embed_digest_in_image(img_uint8, digest, N_patches=8)
        # benign
        attacked = compound_benign(watermarked)
        res,_,_,_,_,_ = verify_semi_fragile(attacked, img_uint8, key=key)
        stats['benign_total'] += 1
        if res == 'PASS':
            stats['benign_pass'] += 1
        # malicious
        donor = donor_images[i % len(donor_images)]
        mal = malicious_splice(watermarked, (donor*255).astype(np.uint8))
        res_m,_,_,_,_,_ = verify_semi_fragile(mal, img_uint8, key=key)
        stats['mal_total'] += 1
        if res_m == 'TAMPER':
            stats['mal_pass'] += 1
    return stats

# run small eval using skimage images
images = [img]
donors = [donor]
stats = evaluate_on_dataset(images, donors)
print(stats)

# %%
# Save results and plots (optional)
plt.savefig('semi_fragile_demo.png', bbox_inches='tight')
print('Saved demo plot to semi_fragile_demo.png')

# PyTorch training loop: Residual encoder/decoder for semi-fragile watermarking
# -------------------------------------------------
# Overview:
# This section adds a PyTorch training loop that learns a residual encoder (U-Net style)
# and a decoder to extract a short payload (bitstring). The encoder outputs a small residual
# added to the cover image. During training, images pass through a differentiable attack pipeline
# (resize, rotate, gaussian blur, additive noise) and a non-differentiable JPEG simulation.
# Losses: bit BCE, perceptual (VGG feature) loss, and MSE image similarity.

# %%
# Additional imports for training
import random
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# %%
# Simple Dataset wrapper: use a directory of images (place images in ./images_train)
class SimpleImageFolder(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.paths = list(Path(root_dir).glob('**/*.jpg')) + list(Path(root_dir).glob('**/*.png'))
        self.t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = str(self.paths[idx])
        img = io.imread(p)
        if img.ndim==2:
            img = np.stack([img,img,img],axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img/255.0).astype(np.float32)
        img = torch.from_numpy(img).permute(2,0,1)
        img = F.interpolate(img.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
        return img

# %%
# Small U-Net style encoder (produces residual) and decoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden=64):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(nn.Conv2d(hidden, hidden*2, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(hidden*2, hidden*2, 3, padding=1), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(hidden*2, hidden, 2, stride=2), nn.ReLU())
        self.out_conv = nn.Conv2d(hidden, in_channels, 1)
    def forward(self, x):
        d1 = self.down1(x)
        p = self.pool(d1)
        d2 = self.down2(p)
        u = self.up1(d2)
        # skip connection
        r = u + d1
        res = torch.tanh(self.out_conv(r)) * 0.1  # small residual scale
        return res

class Decoder(nn.Module):
    def __init__(self, in_channels=3, payload_len=64, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((16,16)))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(hidden*16*16, 512), nn.ReLU(), nn.Linear(512, payload_len))
    def forward(self, x):
        f = self.conv(x)
        out = self.fc(f)
        return out

# %%
# Differentiable attack pipeline (torch tensors)
class DifferentiableAttack(nn.Module):
    def __init__(self, p_jpeg=0.5):
        super().__init__()
        self.p_jpeg = p_jpeg
    def forward(self, imgs):
        # imgs: [B,3,H,W] in [0,1]
        x = imgs
        # random resize + rescale
        if random.random() < 0.9:
            scales = torch.empty(x.size(0)).uniform_(0.8,1.0).tolist()
            out = torch.zeros_like(x)
            for i,s in enumerate(scales):
                h,w = x.shape[2], x.shape[3]
                nh, nw = max(1,int(h*s)), max(1,int(w*s))
                small = F.interpolate(x[i:i+1], size=(nh,nw), mode='bilinear', align_corners=False)
                back = F.interpolate(small, size=(h,w), mode='bilinear', align_corners=False)
                out[i:i+1] = back
            x = out
        # random rotation small
        if random.random() < 0.5:
            angles = torch.empty(x.size(0)).uniform_(-10,10).tolist()
            grid_list = []
            theta_batch = []
            for ang in angles:
                theta = torch.tensor([[np.cos(np.deg2rad(ang)), -np.sin(np.deg2rad(ang)), 0.0],
                                       [np.sin(np.deg2rad(ang)),  np.cos(np.deg2rad(ang)), 0.0]], dtype=torch.float)
                theta_batch.append(theta.unsqueeze(0))
            theta_batch = torch.cat(theta_batch, dim=0).to(x.device)
            grid = F.affine_grid(theta_batch, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
        # gaussian blur approximated by depthwise conv
        if random.random() < 0.7:
            k = random.choice([1,3,5])
            if k>1:
                kernel = torch.tensor(cv2.getGaussianKernel(k, k/3).astype(np.float32))
                kernel2 = kernel @ kernel.T
                kernel2 = kernel2 / kernel2.sum()
                k_t = kernel2.unsqueeze(0).unsqueeze(0).to(x.device)
                pad = k//2
                out = F.pad(x, (pad,pad,pad,pad), mode='reflect')
                # conv per channel
                out_c = []
                for c in range(3):
                    kc = k_t
                    out_c.append(F.conv2d(out[:,c:c+1,:,:], kc, padding=0))
                x = torch.cat(out_c, dim=1)
        # additive noise
        if random.random() < 0.9:
            noise = torch.randn_like(x) * 0.005
            x = torch.clamp(x + noise, 0, 1)
        # non-diff JPEG: apply with probability p_jpeg using PIL round-trip on CPU
        if random.random() < self.p_jpeg:
            x_cpu = (x.detach().cpu().numpy()*255).astype(np.uint8)
            B,H,W,_ = x_cpu.shape[0], x_cpu.shape[2], x_cpu.shape[3], x_cpu.shape[1]
            out_cpu = []
            for i in range(x_cpu.shape[0]):
                pil = cv2.cvtColor((x_cpu[i].transpose(1,2,0)), cv2.COLOR_RGB2BGR)
                # encode/decode jpeg
                q = random.randint(60,95)
                is_success, enc = cv2.imencode('.jpg', pil, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                dec_rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                out_cpu.append(dec_rgb.astype(np.float32)/255.0)
            x = torch.from_numpy(np.stack(out_cpu, axis=0)).permute(0,3,1,2).to(x.device).float()
        return x

# %%
# Perceptual loss model (VGG)
vgg_loss_model = models.vgg16(pretrained=True).features[:16].to(device).eval()
for p in vgg_loss_model.parameters():
    p.requires_grad = False

def perceptual_loss(x, y):
    # expects [B,3,H,W] in [0,1], convert to vgg input range
    def prep(z):
        z_clamped = torch.clamp(z,0,1)
        # normalize
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(z.device)
        std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(z.device)
        return (z_clamped - mean)/std
    xf = prep(x)
    yf = prep(y)
    f1 = vgg_loss_model(xf)
    f2 = vgg_loss_model(yf)
    return F.mse_loss(f1, f2)

# %%
# Training loop

def train_residual_encoder(root_images='./images_train', epochs=10, batch_size=8, payload_len=64, lr=1e-4, save_every=1):
    dataset = SimpleImageFolder(root_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    enc = Encoder().to(device)
    dec = Decoder(payload_len=payload_len).to(device)
    attack = DifferentiableAttack().to(device)
    optim = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=lr)

    for epoch in range(epochs):
        enc.train(); dec.train()
        epoch_loss = 0.0
        start = time.time()
        for imgs in loader:
            imgs = imgs.to(device).float()
            B = imgs.size(0)
            # random payload per image
            payload = torch.randint(0,2,(B,payload_len)).float().to(device)
            # map payload bits to a noise map (simple injection) - expand to spatial
            payload_map = payload.unsqueeze(-1).unsqueeze(-1)
            payload_map = payload_map.repeat(1,1,imgs.size(2),imgs.size(3))
            # Encoder: produce residual using cover+payload_map concat
            # For simplicity, we concatenate payload as extra channels: average pooling to reduce channels
            # Here we'll tile a compressed payload (first channel) to concat
            pchan = payload.float().unsqueeze(2).unsqueeze(3)
            pchan = pchan.repeat(1,1,imgs.size(2),imgs.size(3))
            # reduce pchan to 1 channel by XOR-ish map (sum mod 1)
            p_comb = (pchan.sum(dim=1, keepdim=True) % 1.0)
            enc_in = imgs
            residual = enc(enc_in)
            watermarked = torch.clamp(imgs + residual, 0.0, 1.0)
            # Attack
            attacked = attack(watermarked)
            # Decoder attempts to recover payload from attacked image
            logits = dec(attacked)
            bce = F.binary_cross_entropy_with_logits(logits, payload)
            mse = F.mse_loss(watermarked, imgs)
            perc = perceptual_loss(watermarked, imgs)
            loss = bce + 0.1*mse + 0.5*perc
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += loss.item()
        dur = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f} - time: {dur:.1f}s")
        # save
        if (epoch+1) % save_every == 0:
            torch.save({'enc':enc.state_dict(), 'dec':dec.state_dict()}, f'residual_checkpoint_epoch{epoch+1}.pt')
    return enc, dec

# %%
# How to run training (example):
# 1. Create a folder named 'images_train' in the Colab working directory and upload ~100-1000 images.
# 2. Run: enc, dec = train_residual_encoder(root_images='./images_train', epochs=10, batch_size=8)

# End of added training section
# -------------------------------------------------
# Original Notes & next steps:
# - This training loop is a starting point. Improvements:
#   * Proper payload embedding mechanism (instead of ad-hoc tiled payload_map) — encode payload into a small learned embedding and concatenate.
#   * Use Reed-Solomon / BCH coding on payload and add ECC loss for decoder to predict soft bits.
#   * Better data augmentation and curriculum: start with mild attacks, then increase severity.
#   * Add validation set and compute BER during training.
#   * Tune weighting of perceptual vs bit losses.
# - If you'd like, I can further:
#   1) integrate the learned residual encoder with the classical DWT+DCT+SVD embedding (hybrid),
#   2) add ECC (Reed-Solomon) and per-patch aggregation, or
#   3) produce a downloadable .ipynb containing the full notebook ready to run on Colab.

# === ERROR CORRECTION, PER-PATCH VOTING & SIGNAL FUSION ===
# Add Reed–Solomon ECC around payload, per-patch weighted voting, and combined fragile+robust verification.

# %%
# Install Reed-Solomon (run in Colab)
!pip install -q reedsolo

# %%
import reedsolo

# %%
# Bit/byte helpers

def bits_to_bytes(bitstr: str) -> bytes:
    # pad to byte boundary
    pad = (-len(bitstr)) % 8
    bitstr_padded = bitstr + ('0'*pad)
    b = bytes(int(bitstr_padded[i:i+8], 2) for i in range(0, len(bitstr_padded), 8))
    return b

def bytes_to_bits(b: bytes, nbits: int=None) -> str:
    s = ''.join(f'{byte:08b}' for byte in b)
    if nbits is None:
        return s
    return s[:nbits]

# %%
# Reed-Solomon wrappers
def rs_encode_bits(bitstr: str, nsym: int=32) -> str:
    """
    Encode a bitstring with Reed-Solomon. Returns encoded bits (as string).
    nsym: number of parity bytes to use (higher -> stronger ECC but longer codeword).
    """
    b = bits_to_bytes(bitstr)
    rsc = reedsolo.RSCodec(nsym)
    enc = rsc.encode(b)
    return bytes_to_bits(enc)


def rs_decode_bits(encoded_bitstr: str, orig_nbits: int, nsym: int=32):
    """
    Decode a Reed-Solomon encoded bitstring back to original bits.
    Returns (decoded_bitstr, success_flag, errors_corrected)
    """
    try:
        enc_bytes = bits_to_bytes(encoded_bitstr)
        rsc = reedsolo.RSCodec(nsym)
        dec, _, _ = rsc.decode(enc_bytes)
        dec_bits = bytes_to_bits(dec, orig_nbits)
        return dec_bits, True, None
    except Exception as e:
        # decoding failed
        return None, False, str(e)

# %%
# Update embedding routine to accept ECC encoded digest

def embed_digest_with_rs(rgb_img, digest_bits, nsym=32, **kwargs):
    enc_bits = rs_encode_bits(digest_bits, nsym=nsym)
    return embed_digest_in_image(rgb_img, enc_bits, **kwargs), enc_bits

# %%
# Per-patch extraction that returns confidence as well

def extract_digest_with_confidence(rgb_img, N_patches=8, patch_size=64, bits_per_patch=16, alpha=0.02):
    kp_list = get_sift_keypoints(rgb_img, max_kp=N_patches*4)
    chosen = kp_list[:N_patches]
    extracted_bits = []
    confidences = []
    for i,kp in enumerate(chosen):
        patch, (x1,y1,x2,y2) = extract_patch(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), kp, patch_size)
        Yp = to_y_channel(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        LL, (LH,HL,HH) = dwt2_channel(Yp)
        dct_LL = block_dct(LL)
        U,S,Vt = np.linalg.svd(dct_LL, full_matrices=False)
        med = np.median(S)
        k = min(bits_per_patch, len(S))
        # bit read as sign relative to median
        patch_bits = ''.join('1' if (s-med)>0 else '0' for s in S[:k])
        # confidence heuristic: how dominant are the top singular values compared to median
        top_mean = float(np.mean(np.abs(S[:k]-med)))
        denom = float(np.mean(np.abs(S))+1e-9)
        conf = 1.0 - np.tanh(top_mean/denom)  # ranges (0,1], higher means S are close to median -> low perturbation
        # invert so larger conf means more reliable (we may want smaller top_mean -> more stable)
        confidences.append(float(conf))
        extracted_bits.append(patch_bits)
    return extracted_bits, confidences

# %%
# Weighted majority voting across patches

def weighted_majority_aggregate(extracted_bits_list, confidences, expected_len=None):
    """
    extracted_bits_list: list of strings (per-patch bitstrings, may be shorter)
    confidences: list of floats same length
    expected_len: expected total length of final bitstring (if None, computed)
    returns: aggregated_bitstring, avg_confidence
    """
    if len(extracted_bits_list)==0:
        return '', 0.0
    # first, ensure all bitstrings are padded to same length
    maxlen = max(len(s) for s in extracted_bits_list)
    if expected_len is None:
        expected_len = maxlen
    votes = np.zeros((expected_len, 2), dtype=float)  # counts for 0 and 1
    for s,conf in zip(extracted_bits_list, confidences):
        s_p = s.ljust(expected_len, '0')[:expected_len]
        for i,ch in enumerate(s_p):
            votes[i, int(ch)] += conf
    final_bits = ''.join('1' if votes[i,1] >= votes[i,0] else '0' for i in range(expected_len))
    avg_conf = float(np.mean(confidences))
    return final_bits, avg_conf

# %%
# RS-aware extraction wrapper: aggregate patches -> RS decode -> return decoded bits and status

def extract_and_decode_rs(rgb_img, orig_nbits=128, N_patches=8, bits_per_patch=16, nsym=32):
    extracted_list, confs = extract_digest_with_confidence(rgb_img, N_patches=N_patches, patch_size=64, bits_per_patch=bits_per_patch)
    agg_bits, avg_conf = weighted_majority_aggregate(extracted_list, confs, expected_len=orig_nbits+nsym*8)
    decoded, ok, info = rs_decode_bits(agg_bits, orig_nbits, nsym=nsym)
    return decoded, ok, info, avg_conf

# %%
# Combine fragile (semi-fragile digest) and robust (ownership watermark) signals
# For demonstration we provide a placeholder robust verification function. In your pipeline,
# replace `robust_verify()` with your actual robust watermark verification routine.

def robust_verify_placeholder(received_img):
    """
    Placeholder returning (owner_id_match:bool, confidence:float).
    Replace with your DWT+DCT+SVD-based robust verification that returns a boolean and a confidence score.
    """
    # Here we simulate: assume robust watermark survives benign attacks, fails for heavy tamper
    # In practice, you should run your robust extraction and compare payload/ECC.
    return True, 0.85


def fuse_decisions(fr_result, fr_conf, robust_ok, robust_conf, thresholds=dict(pass_conf=0.6, tamper_conf=0.3)):
    """
    fr_result: 'PASS' | 'TAMPER' | 'UNCERTAIN'
    fr_conf: float [0,1]
    robust_ok: bool
    robust_conf: float [0,1]

    Fusion rules (simple interpretable logic):
      - If fragile says TAMPER -> return TAMPER (regardless of robust) unless robust strongly confirms owner AND fr_conf very low.
      - If fragile says PASS and robust_ok -> PASS (owner and not tampered).
      - If fragile UNCERTAIN but robust_ok and robust_conf high -> POSSIBLE_PASS.
      - If fragile PASS but robust_ok False -> FLAG_FOR_REVIEW (possible removal of ownership or re-watermarking).
    """
    if fr_result == 'TAMPER':
        if robust_ok and robust_conf > 0.9 and fr_conf < 0.2:
            return 'DISPUTED'  # robust strongly says owner but fragile indicates tamper with low confidence
        return 'TAMPER'
    if fr_result == 'PASS':
        if robust_ok:
            return 'PASS'  # ownership + no tamper
        else:
            return 'PASS_NO_OWNERSHIP'  # image seems unmodified but no robust ownership
    if fr_result == 'UNCERTAIN':
        if robust_ok and robust_conf > 0.7:
            return 'POSSIBLE_PASS'
        elif fr_conf < 0.4 and not robust_ok:
            return 'FLAG_FOR_REVIEW'
        else:
            return 'UNCERTAIN'

# %%
# Example combined verification function using the new ECC + voting + fusion

def combined_verification_pipeline(received_rgb, original_rgb, key=b'my_secret_key',
                                   orig_nbits=128, nsym=32, N_patches=8, bits_per_patch=16):
    # Semi-fragile (fragile) path: extract + RS decode
    decoded, ok, info, avg_conf = extract_and_decode_rs(received_rgb, orig_nbits=orig_nbits, N_patches=N_patches, bits_per_patch=bits_per_patch, nsym=nsym)
    # compute recomputed digest (owner side)
    orig_feat = extract_vgg_descriptor(original_rgb)
    orig_digest = compute_hmac_digest(orig_feat, key=key, digest_bits=orig_nbits)
    # decision
    if ok and decoded == orig_digest:
        fr_result = 'PASS'
    elif ok and decoded is not None:
        # compare hamming
        ham = hamming_distance(decoded, orig_digest)
        if ham <= 12:
            fr_result = 'PASS'
        elif ham > 30:
            fr_result = 'TAMPER'
        else:
            fr_result = 'UNCERTAIN'
    else:
        fr_result = 'UNCERTAIN'
    # fragile confidence is avg_conf
    fr_conf = avg_conf
    # robust verification (placeholder - replace with actual robust pipeline)
    robust_ok, robust_conf = robust_verify_placeholder(received_rgb)
    fused = fuse_decisions(fr_result, fr_conf, robust_ok, robust_conf)
    return {
        'fragile_result': fr_result,
        'fragile_confidence': fr_conf,
        'robust_ok': robust_ok,
        'robust_confidence': robust_conf,
        'fused_decision': fused,
        'rs_info': info
    }

# %%
# Quick example (run after embedding with RS):
# embed with RS
# enc_bits = rs_encode_bits(digest, nsym=32)
# watermarked, _ = embed_digest_in_image(img_uint8, enc_bits, N_patches=8)
# res = combined_verification_pipeline(attacked_benign, img_uint8)
# print(res)

# End of ECC / Voting / Fusion section

# === IMPROVED HYBRID TRAINING & EVALUATION (SUBSET, HIGHER LR, METRICS) ===
# This section replaces/extends the previous training routine with:
# - dataset subsetting (10k train, 2k val, 2k test)
# - hybrid DWT+DCT+SVD embedding retained (embed_digest_in_image)
# - learned residual encoder/decoder (hybrid) with higher LR and scheduler
# - evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PSNR, SSIM, BER
# - attack suite for realistic testing
# - early stopping and checkpointing

# %%
# Additional imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import random
import math

# %%
# Utility: create dataset splits (subset 10k train, 2k val, 2k test)
def create_subsets(root_dir, train_n=10000, val_n=2000, test_n=2000, seed=42):
    paths = list(Path(root_dir).glob('**/*.jpg')) + list(Path(root_dir).glob('**/*.png'))
    random.Random(seed).shuffle(paths)
    total_needed = train_n + val_n + test_n
    if len(paths) < total_needed:
        raise ValueError(f"Not enough images in {root_dir} — found {len(paths)}, need {total_needed}")
    train_paths = paths[:train_n]
    val_paths = paths[train_n:train_n+val_n]
    test_paths = paths[train_n+val_n:train_n+val_n+test_n]
    return train_paths, val_paths, test_paths

class SubsetImageDataset(Dataset):
    def __init__(self, paths, image_size=256):
        self.paths = [str(p) for p in paths]
        self.image_size = image_size
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = io.imread(p)
        if img.ndim==2:
            img = np.stack([img,img,img],axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img/255.0).astype(np.float32)
        # center crop or resize to square
        H,W = img.shape[:2]
        side = min(H,W)
        cy, cx = H//2, W//2
        img_c = img[cy-side//2:cy-side//2+side, cx-side//2:cx-side//2+side]
        img_t = cv2.resize(img_c, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img_t).permute(2,0,1).float()
        return img_t

# %%
# Hybrid model: encoder produces residual; final watermarked image also runs classical embedding per-batch optionally
class HybridSystem:
    def __init__(self, enc, dec, use_classical_embed=True, digest_bits=128, key=b'my_secret_key', nsym=32):
        self.enc = enc
        self.dec = dec
        self.use_classical = use_classical_embed
        self.digest_bits = digest_bits
        self.key = key
        self.nsym = nsym
    def embed_classical_batch(self, imgs_np_batch):
        # imgs_np_batch: [B,H,W,3] uint8
        out_list = []
        for img_np in imgs_np_batch:
            feat = extract_vgg_descriptor(img_np)
            digest = compute_hmac_digest(feat, key=self.key, digest_bits=self.digest_bits)
            enc_bits = rs_encode_bits(digest, nsym=self.nsym)
            wm, _ = embed_digest_in_image(img_np, enc_bits, N_patches=8)
            out_list.append(wm)
        return np.stack(out_list, axis=0)

# %%
# Training + evaluation functions

def train_and_evaluate(root_images, gpu_device='cuda', epochs=6, batch_size=16, lr=3e-3, payload_len=64,
                       train_subset=10000, val_subset=2000, test_subset=2000, early_stop_patience=3):
    device = gpu_device if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    train_paths, val_paths, test_paths = create_subsets(root_images, train_n=train_subset, val_n=val_subset, test_n=test_subset)
    train_ds = SubsetImageDataset(train_paths)
    val_ds = SubsetImageDataset(val_paths)
    test_ds = SubsetImageDataset(test_paths)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Train samples:', len(train_ds), 'Val:', len(val_ds), 'Test:', len(test_ds))

    enc = Encoder().to(device)
    dec = Decoder(payload_len=payload_len).to(device)
    attack = DifferentiableAttack().to(device)
    hybrid = HybridSystem(enc, dec, use_classical_embed=True)

    params = list(enc.parameters()) + list(dec.parameters())
    optim = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, verbose=True)

    best_val = 1e9
    no_improve = 0
    history = {'train_loss':[], 'val_loss':[], 'val_bitacc':[], 'val_precision':[], 'val_recall':[], 'val_f1':[]}

    for epoch in range(epochs):
        enc.train(); dec.train()
        running_loss = 0.0
        running_steps = 0
        t0 = time.time()
        for i, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            B = imgs.size(0)
            # Classical embed applied on CPU per-batch for hybrid effect
            imgs_np = (imgs.permute(0,2,3,1).cpu().numpy()*255.0).astype(np.uint8)
            classical_wm_np = hybrid.embed_classical_batch(imgs_np)
            classical_wm_t = torch.from_numpy(classical_wm_np.astype(np.float32)/255.0).permute(0,3,1,2).to(device)

            # Learned residual on top
            residual = enc(classical_wm_t)
            watermarked = torch.clamp(classical_wm_t + residual, 0.0, 1.0)

            attacked = attack(watermarked)
            logits = dec(attacked)
            # generate random payload per sample
            payload = torch.randint(0,2,(B,payload_len)).float().to(device)
            bce = F.binary_cross_entropy_with_logits(logits, payload)
            mse = F.mse_loss(watermarked, classical_wm_t)
            perc = perceptual_loss(watermarked, classical_wm_t)
            loss = bce + 0.05*mse + 0.3*perc

            optim.zero_grad(); loss.backward(); optim.step()
            running_loss += loss.item(); running_steps += 1

            if (i+1) % 20 == 0:
                # compute bit accuracy for this minibatch
                with torch.no_grad():
                    pred_bits = (torch.sigmoid(logits) > 0.5).float()
                    bit_acc = (pred_bits == payload).float().mean().item()
                print(f"Epoch {epoch+1}/{epochs} Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} BitAcc: {bit_acc*100:.2f}%")

        train_epoch_loss = running_loss / (running_steps + 1e-9)
        history['train_loss'].append(train_epoch_loss)

        # Validation
        enc.eval(); dec.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                B = imgs.size(0)
                imgs_np = (imgs.permute(0,2,3,1).cpu().numpy()*255.0).astype(np.uint8)
                classical_wm_np = hybrid.embed_classical_batch(imgs_np)
                classical_wm_t = torch.from_numpy(classical_wm_np.astype(np.float32)/255.0).permute(0,3,1,2).to(device)
                residual = enc(classical_wm_t)
                watermarked = torch.clamp(classical_wm_t + residual, 0.0, 1.0)
                attacked = attack(watermarked)
                logits = dec(attacked)
                payload = torch.randint(0,2,(B,payload_len)).float().to(device)
                bce = F.binary_cross_entropy_with_logits(logits, payload)
                val_losses.append(bce.item())
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().reshape(-1)
                targs = payload.long().cpu().numpy().reshape(-1)
                all_preds.extend(preds.tolist())
                all_targets.extend(targs.tolist())

        val_loss = np.mean(val_losses)
        # compute metrics
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except Exception:
            auc = 0.5
        history['val_loss'].append(val_loss)
        history['val_bitacc'].append(acc)
        history['val_precision'].append(prec)
        history['val_recall'].append(rec)
        history['val_f1'].append(f1)

        print(f"*** Epoch {epoch+1} Summary: TrainLoss: {train_epoch_loss:.4f} ValLoss: {val_loss:.4f} ValAcc: {acc*100:.2f}% Prec: {prec:.3f} Rec: {rec:.3f} F1: {f1:.3f} AUC: {auc:.3f}")

        # scheduler & early stopping
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({'enc':enc.state_dict(), 'dec':dec.state_dict()}, 'best_residual_hybrid.pt')
            print('Saved best model')
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print('Early stopping triggered')
                break

    # After training: evaluate on test set with attack suite
    print('Running final test-suite evaluation...')
    test_metrics = evaluate_test_suite(test_loader, hybrid, enc, dec, attack, device)

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss curves')
    plt.subplot(1,2,2)
    plt.plot(history['val_bitacc'], label='val_bitacc')
    plt.plot(history['val_f1'], label='val_f1')
    plt.legend()
    plt.title('Validation metrics')
    plt.show()

    return enc, dec, history, test_metrics

# %%
# Test-suite evaluation: multiple attacks -> compute BER, PSNR, SSIM, detection metrics

def evaluate_test_suite(test_loader, hybrid, enc, dec, attack_model, device):
    enc.eval(); dec.eval()
    results = []
    attacks = {
        'benign': lambda x: compound_benign(x),
        'jpeg_q50': lambda x: attack_jpeg(x, quality=50),
        'resize_0.7': lambda x: attack_resize(x, scale=0.7),
        'blur_k5': lambda x: attack_blur(x, ksize=5),
        'rotate_15': lambda x: attack_rotate(x, angle=15)
    }
    all_summary = {}
    with torch.no_grad():
        for atk_name, atk_fn in attacks.items():
            ber_list = []
            psnr_list = []
            ssim_list = []
            bit_accs = []
            for imgs in tqdm(test_loader, desc=f'Test {atk_name}'):
                imgs = imgs.to(device)
                B = imgs.size(0)
                imgs_np = (imgs.permute(0,2,3,1).cpu().numpy()*255.0).astype(np.uint8)
                classical_wm_np = hybrid.embed_classical_batch(imgs_np)
                classical_wm_t = torch.from_numpy(classical_wm_np.astype(np.float32)/255.0).permute(0,3,1,2).to(device)
                residual = enc(classical_wm_t)
                watermarked = torch.clamp(classical_wm_t + residual, 0.0, 1.0)

                # apply attack in numpy domain for classical sim
                watermarked_np = (watermarked.permute(0,2,3,1).cpu().numpy()*255.0).astype(np.uint8)
                attacked_np = np.stack([atk_fn(img) for img in watermarked_np], axis=0)
                attacked_t = torch.from_numpy(attacked_np.astype(np.float32)/255.0).permute(0,3,1,2).to(device)

                logits = dec(attacked_t)
                payload = torch.randint(0,2,(B,dec.fc[-1].out_features if hasattr(dec.fc[-1],'out_features') else payload_len)).float().to(device)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().reshape(-1)
                targs = payload.long().cpu().numpy().reshape(-1)
                # BER
                ber = np.mean(preds != targs)
                bit_accs.append(1.0 - ber)
                # PSNR/SSIM between watermarked and attacked
                for i in range(B):
                    wm = watermarked_np[i]
                    at = attacked_np[i]
                    psnr_list.append(psnr(wm, at))
                    try:
                        ssim_list.append(ssim(wm, at, multichannel=True))
                    except Exception:
                        ssim_list.append(0.0)
                ber_list.append(ber)

            all_summary[atk_name] = {
                'mean_bit_acc': float(np.mean(bit_accs)),
                'mean_psnr': float(np.mean(psnr_list)),
                'mean_ssim': float(np.mean(ssim_list)),
            }
            print(f"Attack {atk_name}: BitAcc: {all_summary[atk_name]['mean_bit_acc']:.3f}, PSNR: {all_summary[atk_name]['mean_psnr']:.2f}, SSIM: {all_summary[atk_name]['mean_ssim']:.3f}")
    return all_summary

# %%
# How to run (summary):
# 1. Mount drive and set ROOT_IMAGES to your VOC JPEGImages folder
#    ROOT_IMAGES = '/content/drive/MyDrive/project_codes/models_new/JPEGImages'
# 2. Run: enc, dec, history, test_metrics = train_and_evaluate(ROOT_IMAGES, epochs=6, batch_size=16, lr=3e-3)

# End of improved section

# End of notebook
# -------------------------------------------------
# Notes & next steps:
# - This improved flow aims for higher accuracy and faster experimentation by using 10k train subset and
#   increased learning rate. It preserves the classical DWT+DCT+SVD embedder and stacks a learned residual.
# - Tweak payload_len, batch_size, and attack intensity to balance robustness vs invisibility.
