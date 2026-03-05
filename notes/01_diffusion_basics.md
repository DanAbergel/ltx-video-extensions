# Diffusion Models - Fundamentals

## What is a Diffusion Model?

A generative model that learns to create data (images, video) by learning to **reverse a noise process**.

## Two Processes

### 1. Forward Process (Adding Noise)
- Take a clean image x0
- Gradually add Gaussian noise over T steps: x0 -> x1 -> x2 -> ... -> xT
- At step T, the image is pure noise
- This is FIXED (not learned) - just math

```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
```

where alpha_t follows a **noise schedule** (starts near 1, ends near 0)

### 2. Reverse Process (Denoising) - THIS IS WHAT THE MODEL LEARNS
- Start from pure noise xT
- Learn to predict the noise that was added at each step
- Remove predicted noise step by step: xT -> xT-1 -> ... -> x0
- The neural network (U-Net or DiT) predicts: epsilon_theta(x_t, t)

```
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta(x_t, t)) + sigma_t * z
```

## Key Concepts

### Noise Schedule
- Controls how fast noise is added
- Linear, cosine, or learned schedules
- **beta_t**: noise variance at step t (small, e.g., 0.0001 to 0.02)
- **alpha_t = 1 - beta_t**: signal retention
- **alpha_bar_t = product of all alpha_s for s=1..t**: cumulative signal

### Training Objective
The model minimizes:
```
L = E[||epsilon - epsilon_theta(x_t, t)||^2]
```
= "predict the noise that was added to x0 to get x_t"

### Classifier-Free Guidance (CFG)
- Train model both with and without text conditioning
- At inference: amplify the conditional signal
```
epsilon_guided = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
```
- Higher guidance_scale = follows prompt more closely (but less diverse)

## Latent Diffusion (Stable Diffusion / LTX-Video)

Instead of diffusing in pixel space (expensive), work in **latent space**:

1. **Encode**: image -> VAE encoder -> small latent (e.g., 64x64 instead of 512x512)
2. **Diffuse**: add/remove noise in this compact latent space
3. **Decode**: clean latent -> VAE decoder -> image

Benefits:
- 48x less computation (latent is much smaller)
- Same quality

For video: the VAE is a **Causal 3D VAE** that encodes spatial + temporal dimensions.

## From U-Net to DiT (Diffusion Transformer)

Traditional diffusion models use U-Net (conv + skip connections).

DiT replaces U-Net with a **Transformer**:
- Input: patchified latent tokens (like ViT)
- Conditioning (text, timestep) via **adaptive layer norm (adaLN)** or cross-attention
- Output: predicted noise (same shape as input)

Why better?
- Transformers scale better with more compute/data
- Global attention (every patch sees every other patch)
- More flexible conditioning

## Resources
- Illustrated Stable Diffusion: https://jalammar.github.io/illustrated-stable-diffusion/
- Lilian Weng blog: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- DDPM paper: https://arxiv.org/abs/2006.11239
