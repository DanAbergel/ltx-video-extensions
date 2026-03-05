#!/usr/bin/env python3
"""
Explore LTX-Video architecture - understand the model before extending it.

Run after setup. This script loads the model and prints:
1. Model architecture (all layers and their shapes)
2. Where conditioning happens (cross-attention layers)
3. Memory usage
4. Identifies extension points for adding new controls

Usage:
    python scripts/02_explore_architecture.py
"""

import sys
from pathlib import Path

# Add LTX-Video to path
LTX_DIR = Path(__file__).parent.parent / "LTX-Video"
sys.path.insert(0, str(LTX_DIR))


def explore_model():
    """Load and explore the DiT model architecture."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    except ImportError:
        print("PyTorch not installed. Run setup first.")
        return

    print("\n" + "=" * 60)
    print("  Exploring LTX-Video Architecture")
    print("=" * 60)

    # Try to load and inspect the model
    try:
        from ltx_video.inference import load_pipeline
        print("\n[1] Loading pipeline config...")
        # Will need to adapt this based on actual LTX-Video API
        print("TODO: Load model and inspect layers")
        print("See LTX-Video/ltx_video/ source code for model definition")
    except ImportError:
        print("\nLTX-Video not installed yet. Run 01_setup_and_run.sh first.")
        print("In the meantime, let's explore what we know:")

    print("\n" + "-" * 60)
    print("  LTX-Video DiT Architecture (from paper)")
    print("-" * 60)
    print("""
    Input Pipeline:
    ===============
    Text Prompt -> Gemma-3 Encoder -> text_embeddings [seq_len, dim]
    Video/Noise -> Causal 3D VAE Encoder -> latent [B, C, T, H, W]
    Latent -> Patchify -> tokens [B, num_patches, dim]

    DiT Blocks (repeated N times):
    ===============================
    For each block:
      1. adaLN(tokens, timestep_embedding)     # Adaptive LayerNorm with timestep
      2. self_attention(tokens)                  # All patches attend to all patches
      3. cross_attention(tokens, text_embeddings) # <-- WHERE TEXT CONDITIONING HAPPENS
      4. adaLN(tokens, timestep_embedding)
      5. feedforward(tokens)

    Output:
    =======
    tokens -> Unpatchify -> predicted_noise [B, C, T, H, W]
    Clean latent -> Causal 3D VAE Decoder -> Video [B, 3, T, H, W]

    EXTENSION POINTS for new controls:
    ===================================
    A) Cross-attention: add new embeddings alongside text_embeddings
       -> e.g., depth map encoded by a small network
       -> Easiest: concat new tokens to text tokens before cross-attention

    B) Channel concatenation: add extra channels to latent input
       -> e.g., depth latent concatenated to noise latent
       -> Need to modify first linear layer (more input channels)

    C) ControlNet approach: parallel DiT copy
       -> Heavy but proven approach (used in Stable Diffusion)
       -> Copy DiT, freeze original, train copy on new condition
       -> Copy's outputs are ADDED to original's intermediate features

    D) adaLN injection: add new conditioning signal to timestep embedding
       -> e.g., motion_intensity scalar -> MLP -> add to timestep embedding
       -> Simplest for scalar controls
    """)

    print("\n" + "-" * 60)
    print("  Key Files to Read in LTX-Video/ltx_video/")
    print("-" * 60)
    print("""
    Start with these files (read in order):
    1. inference.py            - Entry point, understand the full pipeline
    2. ltx_video/pipelines/    - Pipeline orchestration (scheduler, VAE, DiT)
    3. ltx_video/models/       - DiT model definition (transformer blocks)
    4. ltx_video/models/transformers/ - The actual DiT architecture
    5. configs/*.yaml          - Model configs (dimensions, num layers, etc.)
    """)


if __name__ == "__main__":
    explore_model()
