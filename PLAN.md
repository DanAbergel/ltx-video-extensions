# LTX-Video Extensions - Project Plan

## Goal
Extend LTX-Video (Lightricks open-source DiT model) with new controls/use cases.
Build portfolio evidence for Lightricks Applications Research Scientist role.

## Timeline: ~2 weeks

---

## Phase 1: Foundations (Days 1-3)

### Day 1: Theory - Diffusion Models
- [ ] Read: "The Illustrated Stable Diffusion" (Jay Alammar)
- [ ] Read: DDPM paper summary (Lilian Weng blog: "What are Diffusion Models?")
- [ ] Understand: forward process (add noise), reverse process (denoise), noise scheduling
- [ ] Understand: latent diffusion (encode to latent space, diffuse there, decode back)
- [ ] Take notes in `notes/01_diffusion_basics.md`

### Day 2: Theory - Diffusion Transformers (DiT)
- [ ] Read: DiT paper (Peebles & Xie, "Scalable Diffusion Models with Transformers")
- [ ] Read: LTX-2 paper on arxiv: https://arxiv.org/html/2601.03233v1
- [ ] Understand: how DiT replaces U-Net with a Transformer in latent space
- [ ] Understand: LTX-Video specifics: causal VAE, Gemma-3 text encoder, dual-stream audio/video
- [ ] Take notes in `notes/02_dit_architecture.md`

### Day 3: Setup + Run LTX-Video
- [ ] Clone repo: `git clone https://github.com/Lightricks/LTX-Video.git`
- [ ] Setup environment (Python 3.10+, CUDA 12.2+, PyTorch 2.1.2+)
- [ ] Download model weights from HuggingFace (start with ltxv-2b-0.9.8-distilled for less VRAM)
- [ ] Run text-to-video inference:
      ```
      python inference.py --prompt "A cat walking on a beach at sunset" \
        --height 480 --width 704 --num_frames 65 \
        --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml
      ```
- [ ] Run image-to-video inference with a conditioning image
- [ ] Document results in `notes/03_first_runs.md`

---

## Phase 2: Understand the Code (Days 4-5)

### Day 4: Dive into LTX-Video source code
- [ ] Read `ltx_video/` source - understand the pipeline structure
- [ ] Map the inference flow: text encoding -> latent noise -> DiT denoising -> VAE decode
- [ ] Understand the config system (YAML files in `configs/`)
- [ ] Identify where conditioning happens (how image-to-video works)
- [ ] Document architecture in `notes/04_code_walkthrough.md`

### Day 5: Understand conditioning mechanisms
- [ ] How does text conditioning work? (Gemma-3 embeddings -> cross-attention)
- [ ] How does image conditioning work? (encode image to latent, concat/replace)
- [ ] How does multi-keyframe conditioning work?
- [ ] Identify extension points for adding new controls
- [ ] Document in `notes/05_conditioning_deep_dive.md`

---

## Phase 3: Implement an Extension (Days 6-10)

### Choose ONE of these (pick based on what feels achievable):

#### Option A: Depth-Conditioned Video Generation
- Add depth map conditioning to guide the spatial structure of generated videos
- Use a pre-trained depth estimator (MiDaS/DPT) to extract depth from reference frames
- Inject depth information into the DiT pipeline as additional conditioning
- This is like ControlNet for video - very relevant to Lightricks' "novel controls"

#### Option B: Style Transfer for Video Generation
- Condition generation on a reference style image (not content, just style)
- Extract style features using CLIP or a VGG-based style encoder
- Inject style embeddings alongside text embeddings in cross-attention
- Generate videos that match a given artistic style

#### Option C: Motion Intensity Control
- Add a simple scalar control for motion intensity (0 = almost still, 1 = fast motion)
- Modify the noise scheduling or conditioning to control temporal change
- Simpler than A/B but still a novel control mechanism

### Implementation steps (any option):
- [ ] Create a branch: `git checkout -b feature/[option-name]`
- [ ] Implement the conditioning encoder
- [ ] Modify the DiT pipeline to accept the new condition
- [ ] Run experiments with different inputs
- [ ] Generate comparison videos (with/without the new control)
- [ ] Document in `notes/06_implementation.md`

---

## Phase 4: Polish & Document (Days 11-14)

- [ ] Write a clear README.md with:
  - What you built and why
  - Architecture diagram (how your extension fits into LTX-Video)
  - Example outputs (GIFs/videos)
  - How to reproduce
- [ ] Clean up code, add docstrings
- [ ] Push to GitHub (public repo)
- [ ] Optional: write a short blog post / Twitter thread

---

## Resources

### Papers
- DDPM: https://arxiv.org/abs/2006.11239
- DiT: https://arxiv.org/abs/2212.09748
- LTX-2: https://arxiv.org/html/2601.03233v1
- ControlNet: https://arxiv.org/abs/2302.05543

### Blogs & Tutorials
- Jay Alammar - The Illustrated Stable Diffusion
- Lilian Weng - What are Diffusion Models?
- HuggingFace Diffusers documentation

### Code
- LTX-Video repo: https://github.com/Lightricks/LTX-Video
- LTX-Video weights: https://huggingface.co/Lightricks/LTX-Video
- ControlNet reference: https://github.com/lllyasviel/ControlNet

---

## Hardware Notes
- Minimum: GPU with 8GB VRAM (use ltxv-2b-0.9.8-distilled + FP8)
- Recommended: GPU with 24GB+ VRAM
- Alternative: Use HUJI SLURM cluster (like you did for FinAgent)
