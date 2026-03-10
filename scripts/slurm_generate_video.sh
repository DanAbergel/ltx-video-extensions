#!/bin/bash
# =====================================================================
# SLURM Job Script — LTX-Video Long Video Generation
# =====================================================================
#
# HOW TO USE:
#   sbatch scripts/slurm_generate_video.sh
#
# MONITOR:
#   squeue -u $USER                              # check job status
#   tail -f logs/ltx_video_<JOB_ID>.out          # watch generation output
#   scancel <JOB_ID>                             # cancel if needed
#
# FIRST TIME SETUP (run once on the login node BEFORE sbatch):
#   cd /sci/labs/arieljaffe/dan.abergel1
#   mkdir -p repos && cd repos
#   git clone https://github.com/DanAbergel/ltx-video-extensions.git
#   python3 -m venv /sci/labs/arieljaffe/dan.abergel1/ltx_env
#   source /sci/labs/arieljaffe/dan.abergel1/ltx_env/bin/activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install diffusers transformers accelerate imageio[pyav] tqdm Pillow
# =====================================================================

# ── SLURM resource request ─────────────────────────────────────────
#SBATCH --job-name=ltx-video-5min
#SBATCH --gres=gpu:l40s:1        # 1x NVIDIA L40S (48 GB VRAM)
#SBATCH --cpus-per-task=8        # 8 CPU cores
#SBATCH --mem=64G                # 64 GB RAM
#SBATCH --time=12:00:00          # 12h max (113 chunks × ~2-4 min/chunk)
#SBATCH --output=logs/ltx_video_%j.out
#SBATCH --error=logs/ltx_video_%j.err

set -euo pipefail

# ── Paths (adapt to your cluster) ──────────────────────────────────
LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/ltx-video-extensions"
VENV_DIR="$LAB_DIR/torch_env"

# ── Redirect caches to lab storage ─────────────────────────────────
# Home directory on Moriah has ~5 GB quota. HuggingFace models are
# huge (several GB each), so redirect everything to lab storage.
export HF_HOME="$LAB_DIR/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$LAB_DIR/cache/huggingface/hub"
export TMPDIR="$LAB_DIR/tmp"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export TRITON_CACHE_DIR="$LAB_DIR/cache/triton"
export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "============================================================"
echo "  LTX-Video 5-Minute Generation — SLURM Job"
echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  Project:    $PROJECT_DIR"
echo "  Venv:       $VENV_DIR"
echo "============================================================"
echo ""

# ── 1. Activate virtual environment ────────────────────────────────
echo "[1/5] Activating venv ..."
source "$VENV_DIR/bin/activate"
echo "  Python: $(which python3)"
echo "  Version: $(python3 --version)"
echo ""

# ── 2. Update code from GitHub ─────────────────────────────────────
echo "[2/5] Updating code ..."
cd "$PROJECT_DIR"
git fetch --all
git reset --hard origin/main
echo "  Commit: $(git rev-parse --short HEAD)"
echo "  Message: $(git log -1 --pretty=%s)"
echo ""

# ── 3. GPU check ───────────────────────────────────────────────────
echo "[3/5] GPU check ..."
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')
"
echo ""

# ── 4. Create output directory & run generation ────────────────────
echo "[4/5] Starting video generation ..."
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/videos"

python3 -u "$PROJECT_DIR/scripts/generate_long_video.py" \
    --prompt "A woman with long brown hair walks through a sunlit garden, flowers blooming around her, golden hour lighting, cinematic" \
    --duration_seconds 300 \
    --fps 24 \
    --height 480 \
    --width 704 \
    --output videos/5min_garden_walk.mp4 \
    --seed 42

# ── 5. Done ─────────────────────────────────────────────────────────
echo ""
echo "[5/5] Done!"
echo "============================================================"
echo "  Job finished:  $(date)"
echo "  Output video:  $PROJECT_DIR/videos/5min_garden_walk.mp4"
echo "  Output JSON:   $PROJECT_DIR/videos/5min_garden_walk.json"
echo "  Output log:    $PROJECT_DIR/videos/5min_garden_walk.log"
echo ""
echo "  Next: git add videos/ && git commit && git push"
echo "============================================================"
