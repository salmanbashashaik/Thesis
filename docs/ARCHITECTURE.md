# Architecture Documentation

## Overview

ALDM consists of two main components:
1. **3D Variational Autoencoder (VAE)**: Compresses MRI volumes into latent space
2. **Conditional Latent Diffusion Model**: Generates synthetic MRI in latent space

---

## 3D Variational Autoencoder (VAE)

### Purpose
- Learn compact anatomical representation from source domain (GBM)
- Compress 3D MRI volumes: (3, 112, 112, 112) → (8, 28, 28, 28)
- Reduce computational cost for diffusion training

### Encoder Architecture

```
Input: (3, 112, 112, 112)  # T1, T2, FLAIR

Conv3D(3 → 64, kernel=3, stride=1, padding=1)
InstanceNorm3D + SiLU

Conv3D(64 → 64, kernel=3, stride=2, padding=1)  # 112 → 56
InstanceNorm3D + SiLU

Conv3D(64 → 128, kernel=3, stride=1, padding=1)
InstanceNorm3D + SiLU

Conv3D(128 → 128, kernel=3, stride=2, padding=1)  # 56 → 28
InstanceNorm3D + SiLU

Conv3D(128 → 128, kernel=3, stride=1, padding=1)  # Refinement
InstanceNorm3D + SiLU

Conv3D(128 → 16, kernel=1)  # μ and log(σ²)

Output: μ, log(σ²) ∈ (8, 28, 28, 28)
```

**Key Features**:
- Strided convolutions for downsampling (factor of 4)
- Instance normalization for batch size = 1
- Refinement blocks at latent resolution
- Reparameterization trick: z = μ + σ ⊙ ε

### Decoder Architecture

```
Input: z ∈ (8, 28, 28, 28)

Conv3D(8 → 128, kernel=3, padding=1)
InstanceNorm3D + SiLU

Conv3D(128 → 128, kernel=3, padding=1)  # Refinement
InstanceNorm3D + SiLU

Upsample(scale=2, mode='trilinear')  # 28 → 56
Conv3D(128 → 128, kernel=3, padding=1)
InstanceNorm3D + SiLU

Conv3D(128 → 64, kernel=3, padding=1)
InstanceNorm3D + SiLU

Upsample(scale=2, mode='trilinear')  # 56 → 112
Conv3D(64 → 64, kernel=3, padding=1)
InstanceNorm3D + SiLU

Conv3D(64 → 3, kernel=3, padding=1)
Tanh()  # Output range: [-1, 1]

Output: (3, 112, 112, 112)
```

**Key Features**:
- Trilinear upsampling (avoids checkerboard artifacts)
- Symmetric architecture to encoder
- Tanh activation for normalized output

### VAE Loss Function

```python
L_VAE = λ_rec * L_rec + λ_KL * L_KL + λ_grad * L_grad

L_rec = ||x - x̂||₁  # L1 reconstruction loss

L_KL = KL(q(z|x) || N(0, I))  # KL divergence

L_grad = Σ_axis ||∇_axis(x̂) - ∇_axis(x)||₁  # Gradient consistency
```

**Hyperparameters**:
- λ_rec = 1.0
- λ_KL = 1e-4 (with warmup)
- λ_grad = 0.1

**KL Warmup**:
```python
λ_KL(t) = λ_KL * min(1, t / (0.35 * T))
```

---

## Latent Normalization

### Purpose
Stabilize diffusion training across subjects with varying intensity distributions.

### Computation

```python
# Compute statistics from GBM training set
μ_latent = E[z]  # Mean across 400 batches
σ_latent = Std[z]  # Std across 400 batches

# Normalize
z_norm = (z - μ_latent) / (σ_latent + ε)

# Denormalize (during decoding)
z = z_norm * (σ_latent + ε) + μ_latent
```

**Parameters**:
- ε = 1e-6 (numerical stability)
- Statistics computed per channel (8 channels)

---

## Conditional Latent Diffusion Model

### Forward Diffusion Process

Gradually add Gaussian noise to latent representation:

```python
z_t = √(ᾱ_t) * z_0 + √(1 - ᾱ_t) * ε

where:
  ε ~ N(0, I)
  β_t ∈ [1e-4, 2e-2]  # Linear schedule
  α_t = 1 - β_t
  ᾱ_t = ∏_{i=1}^t α_i
  T = 1000 timesteps
```

### Reverse Denoising Process

Iteratively denoise from pure noise to clean latent:

```python
z_{t-1} = 1/√α_t * (z_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(z_t, t, c)) + σ_t * ε

where:
  ε_θ = U-Net noise predictor
  c = conditioning (tumor mask)
  σ_t = √β_t
```

---

## 3D U-Net Denoiser

### Architecture Overview

```
Input: z_t ∈ (8, 28, 28, 28), t ∈ [0, 999], c ∈ (1, 28, 28, 28)

Encoder:
  Resolution 28: Conv3D(8 → 64) + ResBlock × 2
  ↓ Downsample (28 → 14)
  Resolution 14: Conv3D(64 → 128) + ResBlock × 2 + Attention
  ↓ Downsample (14 → 7)
  Resolution 7: Conv3D(128 → 256) + ResBlock × 2 + Attention

Bottleneck:
  ResBlock × 2 + Attention

Decoder:
  ↑ Upsample (7 → 14)
  Resolution 14: Conv3D(256+128 → 128) + ResBlock × 2 + Attention
  ↑ Upsample (14 → 28)
  Resolution 28: Conv3D(128+64 → 64) + ResBlock × 2

Output: Conv3D(64 → 8)  # Predicted noise
```

### Residual Block

```python
class ResBlock3D(nn.Module):
    def __init__(self, channels, time_emb_dim):
        self.conv1 = Conv3D(channels, channels, 3, padding=1)
        self.norm1 = GroupNorm(8, channels)
        self.conv2 = Conv3D(channels, channels, 3, padding=1)
        self.norm2 = GroupNorm(8, channels)
        self.time_mlp = Linear(time_emb_dim, channels)
        
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h
```

### Timestep Embedding

```python
def timestep_embedding(t, dim):
    # Sinusoidal positional encoding
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

# Project to time_emb_dim=256
time_emb = MLP(timestep_embedding(t, 128), 256)
```

### Self-Attention Block

```python
class SelfAttention3D(nn.Module):
    def __init__(self, channels, num_heads=4):
        self.norm = GroupNorm(8, channels)
        self.qkv = Conv3D(channels, channels * 3, 1)
        self.proj = Conv3D(channels, channels, 1)
        self.num_heads = num_heads
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, D*H*W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).reshape(B, C, D, H, W)
        return x + self.proj(out)
```

---

## Anatomical Conditioning

### ControlNet Architecture

Processes tumor masks to provide spatial guidance:

```
Input: mask ∈ (1, 112, 112, 112)

# Downsample to latent resolution
Conv3D(1 → 32, stride=2)  # 112 → 56
Conv3D(32 → 64, stride=2)  # 56 → 28

# Process at each U-Net resolution
ResBlock(64) → inject at resolution 28
Downsample → ResBlock(128) → inject at resolution 14
Downsample → ResBlock(256) → inject at resolution 7
```

### Conditioning Mechanisms

**1. FiLM-style Modulation**:
```python
# Global mask statistics
mask_stats = global_pool(mask)  # (B, 1)
γ, β = MLP(mask_stats)  # (B, C)

# Apply to feature maps
h' = h ⊙ (1 + 0.1 * γ) + 0.1 * β
```

**2. ControlNet Injection**:
```python
# At each resolution
control_features = ControlNet(mask, resolution)
unet_features = unet_features + λ_ctrl * control_features
```

**3. Edge and Distance Conditioning**:
```python
# Derive additional signals from mask
edges = sobel_filter(mask)
distance = distance_transform(mask)

# Concatenate with mask
c = concat([mask, edges, distance])  # (3, 28, 28, 28)
```

---

## Classifier-Free Guidance

### Training

Randomly drop conditioning with probability p=0.1:

```python
if random() < 0.1:
    c = ∅  # Null conditioning (zeros)
else:
    c = mask
    
loss = ||ε - ε_θ(z_t, t, c)||²
```

### Inference

Interpolate between conditional and unconditional predictions:

```python
ε_guided = ε_θ(z_t, t, ∅) + s * (ε_θ(z_t, t, c) - ε_θ(z_t, t, ∅))

where s = 3.0 (guidance scale)
```

**Effect**:
- s = 0: Unconditional generation
- s = 1: Standard conditional generation
- s > 1: Stronger adherence to conditioning

---

## Tumor-Weighted Loss

Emphasize tumor regions during training:

```python
# Create weight map
w = 1.0 + (mask > 0) * 1.0  # 2x weight in tumor
w = dilate(w, kernel=3x3x3)  # Include boundary

# Weighted diffusion loss
L_diff = E_t,ε [w * ||ε - ε_θ(z_t, t, c)||²]
```

---

## Exponential Moving Average (EMA)

Stabilize sampling by maintaining averaged weights:

```python
# During training
θ_EMA ← γ * θ_EMA + (1 - γ) * θ

where γ = 0.999
```

**Usage**: EMA weights used for inference, not training.

---

## End-to-End Generation Pipeline

```python
# 1. Sample noise
z_T ~ N(0, I)  # (8, 28, 28, 28)

# 2. Prepare conditioning
c = process_mask(tumor_mask)  # (3, 28, 28, 28)

# 3. Iterative denoising
for t in reversed(range(T)):
    ε_pred = unet_ema(z_t, t, c, guidance_scale=3.0)
    z_{t-1} = denoise_step(z_t, ε_pred, t)

# 4. Denormalize latent
z_0 = z_0 * σ_latent + μ_latent

# 5. Decode to image space
x_syn = vae_decoder(z_0)  # (3, 112, 112, 112)
```

---

## Model Parameters

### VAE
- Encoder: ~15M parameters
- Decoder: ~15M parameters
- Total: ~30M parameters

### Diffusion U-Net
- Encoder: ~40M parameters
- Decoder: ~40M parameters
- Attention: ~10M parameters
- Total: ~90M parameters

### ControlNet
- ~20M parameters

**Total ALDM**: ~140M parameters

---

## Memory Requirements

### Training
- VAE: ~12GB VRAM (batch_size=1)
- Diffusion: ~20GB VRAM (batch_size=1)
- Peak: ~24GB VRAM

### Inference
- VAE: ~4GB VRAM
- Diffusion: ~8GB VRAM
- Total: ~12GB VRAM

---

## Computational Complexity

### VAE Forward Pass
- Encoder: O(C × D × H × W)
- Decoder: O(C × D × H × W)
- Total: ~0.5 GFLOPS

### Diffusion Sampling
- Per timestep: ~2 GFLOPS
- Total (1000 steps): ~2 TFLOPS
- Time: ~5 minutes per sample (A100)

---

## Design Choices

### Why Latent Diffusion?
- 16× reduction in spatial dimensions
- 256× reduction in computational cost
- Preserves anatomical structure

### Why ControlNet?
- Explicit spatial control
- Preserves tumor location and shape
- Flexible conditioning strength

### Why Classifier-Free Guidance?
- No separate classifier needed
- Adjustable conditioning strength at inference
- Better sample quality

---

## References

- VAE: Kingma & Welling (2014)
- DDPM: Ho et al. (2020)
- Latent Diffusion: Rombach et al. (2022)
- ControlNet: Zhang et al. (2023)
- Classifier-Free Guidance: Ho & Salimans (2022)
