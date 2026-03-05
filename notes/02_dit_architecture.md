# Diffusion Transformers (DiT) & LTX-Video Architecture

## DiT: Scalable Diffusion Models with Transformers

Paper: Peebles & Xie (2022) - https://arxiv.org/abs/2212.09748

### Core Idea
Replace U-Net with a Transformer. The denoising network is now a ViT-like architecture.

### How DiT Works

1. **Patchify**: divide noisy latent into non-overlapping patches (like ViT)
   - Latent: 32x32x4 -> patches of 2x2x4 -> 256 tokens of dim 4*2*2=16
   - Linear projection to model dimension d

2. **Add positional embeddings**: sinusoidal or learned

3. **Transformer blocks** with conditioning:
   - Each block: LayerNorm -> Self-Attention -> LayerNorm -> FFN
   - Conditioning via **adaLN-Zero**: adaptive layer norm
     - timestep t + class label c -> MLP -> (gamma, beta, alpha) per layer
     - LayerNorm is modulated: gamma * LN(x) + beta
     - Output is scaled by alpha (initialized to 0 = identity at start)

4. **Final layer**: LayerNorm -> Linear -> predicted noise + predicted variance

### Why adaLN-Zero?
- Tested: in-context conditioning, cross-attention, adaLN, adaLN-Zero
- adaLN-Zero worked best: zero-initialization means the Transformer starts as identity
- Conditioning is "gently" introduced during training

### Scaling
- DiT-S (33M), DiT-B (130M), DiT-L (458M), DiT-XL (675M)
- Larger = better FID, scales like language models
- "Scaling the transformer is more effective than scaling the diffusion steps"

---

## LTX-Video Architecture

### Overview
- DiT-based video generation model
- Operates in 3D latent space (spatial + temporal)
- Text conditioning via Gemma-3 (not CLIP)

### Pipeline

```
Text Prompt --> Gemma-3 Text Encoder --> text embeddings
                                              |
                                              v
Random Noise --> [DiT Denoising Loop] --> Clean Latent --> Causal 3D VAE Decoder --> Video
                  (T denoising steps)

Optional: Image --> Causal 3D VAE Encoder --> latent --> used as first frame conditioning
```

### Key Components

#### 1. Causal 3D VAE
- Encodes video (H x W x T frames) into compact latent (h x w x t x c)
- "Causal" = each frame can only attend to past frames (important for streaming)
- Compression: ~8x spatial, ~4x temporal
- Same VAE for images (single frame = video with T=1)

#### 2. Gemma-3 Text Encoder
- Replaces CLIP (used in older Stable Diffusion)
- Better understanding of nuanced prompts (camera motion, emotions, lighting)
- Dense text embeddings fed via cross-attention to DiT

#### 3. DiT Backbone
- Transformer blocks with:
  - Self-attention over ALL latent tokens (spatial + temporal jointly)
  - Cross-attention to text embeddings
  - Timestep conditioning via adaLN
- This is where the "magic" happens: learning to denoise video latents

#### 4. LTX-2 Dual-Stream (Audio + Video)
- 19B parameters total: 14B video + 5B audio
- Two DiT streams that communicate via bidirectional cross-attention
- Audio stream generates synchronized sound

### Model Variants
- ltxv-13b-0.9.8-dev: highest quality, needs more VRAM
- ltxv-13b-0.9.8-distilled: faster, less VRAM (knowledge distillation)
- ltxv-2b-0.9.8-distilled: smallest, ideal for experimentation
- FP8 versions: quantized for even less memory

### Conditioning Mechanisms (IMPORTANT for our project)
1. **Text**: Gemma-3 embeddings -> cross-attention in every DiT block
2. **Image**: encode with VAE -> replace/concat first frame latent
3. **Multi-keyframe**: multiple images at specified frame positions
4. **Video extension**: generate continuation from existing video latent

### Where to Add New Controls?
- **Cross-attention**: add new conditioning embeddings alongside text
- **Concatenation**: concat new latent channels to noise input
- **adaLN modulation**: add new conditioning signals to the adaptive LayerNorm
- ControlNet approach: parallel DiT that injects features into main DiT

## Resources
- DiT paper: https://arxiv.org/abs/2212.09748
- LTX-2 paper: https://arxiv.org/html/2601.03233v1
- ControlNet paper: https://arxiv.org/abs/2302.05543
