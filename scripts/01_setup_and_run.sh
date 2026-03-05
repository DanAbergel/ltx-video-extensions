#!/bin/bash
# =============================================================
# Step 1: Clone LTX-Video and run first inference
# =============================================================
# Run this on a machine with GPU (HUJI SLURM or local with CUDA)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "  LTX-Video Extensions - Setup"
echo "============================================"

# 1. Clone LTX-Video
if [ ! -d "LTX-Video" ]; then
    echo "[1/4] Cloning LTX-Video..."
    git clone https://github.com/Lightricks/LTX-Video.git
else
    echo "[1/4] LTX-Video already cloned."
fi

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment..."
    python -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "[3/4] Installing LTX-Video..."
cd LTX-Video
pip install -e ".[inference]"
cd ..

# 4. Download smallest model (2B distilled - ~4GB)
echo "[4/4] Downloading model weights..."
echo "The model will be downloaded automatically on first run from HuggingFace."
echo "Model: Lightricks/LTX-Video (ltxv-2b-0.9.8-distilled)"
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next step: Run first inference:"
echo "  cd LTX-Video"
echo '  python inference.py --prompt "A golden retriever playing in the snow" \'
echo '    --height 480 --width 704 --num_frames 65 \'
echo '    --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml'
echo ""
echo "Or for image-to-video:"
echo '  python inference.py --prompt "The scene comes alive with gentle motion" \'
echo '    --conditioning_media_paths /path/to/your/image.jpg \'
echo '    --height 480 --width 704 --num_frames 65 \'
echo '    --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml'
