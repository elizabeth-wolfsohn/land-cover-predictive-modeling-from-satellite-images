#!/usr/bin/env bash
# Prepare a vast.ai instance (PyTorch template): dependencies + dataset from Hugging Face.
# Usage, from the repo root:  export HF_TOKEN=hf_...  &&  bash vast_setup.sh
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN=hf_... first (read token, the dataset is private)}"
HF_REPO=AirFxOnz/Segmentation
ZIP=additional_files_earthcube_emu4zqr.zip
TMP=/workspace/hf_data

# torch / torchvision come with the template image
pip install -q timm tifffile pandas pyyaml tqdm matplotlib "huggingface_hub[cli]"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no GPU visible"
print("torch", torch.__version__, "on", torch.cuda.get_device_name(0))
# raises "no kernel image" if this torch build does not support the GPU (RTX 50xx needs CUDA >= 12.8):
# pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
torch.ones(1, device="cuda") + 1
EOF

if [ ! -d dataset_UNZIPPED ]; then
  hf download "$HF_REPO" "$ZIP" --repo-type dataset --local-dir "$TMP"
  python -m zipfile -e "$TMP/$ZIP" "$TMP"   # creates $TMP/dataset/
  mv "$TMP/dataset" dataset_UNZIPPED
  rm "$TMP/$ZIP"
fi
mkdir -p code/experiments
echo "train images: $(ls dataset_UNZIPPED/train/images | wc -l) (expected 18491)"
echo "test images:  $(ls dataset_UNZIPPED/test/images | wc -l) (expected 5043)"
