#!/usr/bin/env bash
# One-shot environment setup for eeg-channel-adaptation.
#
#   bash setup_env.sh                # creates conda env "eeg-adapt" and installs everything
#   bash setup_env.sh my-env-name    # custom env name
#
# Then:  conda activate eeg-adapt   and run the scripts in scripts/ (see README Quickstart).
set -euo pipefail

ENV_NAME="${1:-eeg-adapt}"
PY_VER="3.12"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
echo "==> Repo:      $REPO_DIR"
echo "==> Conda env: $ENV_NAME (python $PY_VER)"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Create env (idempotent)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Env '$ENV_NAME' already exists; reusing."
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"

# 2. Install the package + all dependencies (braindecode/moabb come from git; takes a few minutes)
echo "==> Installing package + dependencies ..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

# NOTE ON GPU: the line above installs whatever torch braindecode pulls (usually a CUDA wheel).
# If torch does not see your GPU, install the matching build, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Vendored dependency check (OmnEEG method needs this; it is bundled in the repo)
if [ -d vendor/OmnEEG/omneeg ]; then echo "==> vendor/OmnEEG present (OmnEEG method OK)."
else echo "==> WARNING: vendor/OmnEEG missing - the OmnEEG method will not run."; fi

# 4. Local data / results directories (defaults the scripts read/write; all git-ignored)
echo "==> Creating local data/ and results/ directories ..."
mkdir -p data/luna_native data/luna_native_raw data/interpolated data/interpolated_raw \
         data/omneeg data/raw results .cache

# 5. Quick import sanity check
python -c "import torch, braindecode, mne, h5py, torchmetrics, transformers, einops; from adapter_finetuning.optim import CosineAnnealingWarmupLR; print('==> Imports OK. CUDA available:', torch.cuda.is_available())"

cat <<EOF

==> Setup complete. Next:

    conda activate $ENV_NAME

    # 1) preprocess one dataset (BCIC2A auto-downloads via MOABB)
    python scripts/preprocess_luna_native.py --dataset bcic2a

    # 2) quick 1-batch smoke test, then a real run
    python scripts/run_eegpt_experiments.py --mode native --training-mode sft --dataset bcic2a --fast-dev-run
    python scripts/run_eegpt_experiments.py --mode native --training-mode sft --dataset bcic2a --n-seeds 15

EOF
