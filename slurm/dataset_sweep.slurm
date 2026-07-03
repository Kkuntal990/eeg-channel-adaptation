#!/bin/bash
#SBATCH --job-name=ds_sweep
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=CHANGE_ME        # <-- set your SLURM allocation
#SBATCH --output=logs/ds_sweep_%A_%a.out
#SBATCH --error=logs/ds_sweep_%A_%a.err
#SBATCH --array=0-4

REPO="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (portable)
DATA="$REPO/data"
OMNEEG_DATA="$REPO/data/omneeg"

cd $REPO
mkdir -p logs

# --- cluster-specific: adjust module loads for your site ---
module purge 2>/dev/null || true
module load gpu cuda anaconda3 2>/dev/null || true
source activate eeg-adapt   # env created by setup_env.sh

export HF_HOME="$REPO/.cache/huggingface"
export MNE_DATA="$REPO/.cache/mne_data"
export TMPDIR="$REPO/.cache/tmp"
export WANDB_MODE=offline
export PYTHONPATH=$REPO:$PYTHONPATH

DATASETS=(bcic2a physionet tuev faced mdd_mumtaz2016)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "=== $DATASET sweep starting at $(date) ==="

# Test each model with Conv1d SFT on this dataset
for script in run_eegpt_experiments run_cbramod_experiments run_luna_experiments run_neurogpt_experiments; do
    echo "[$script conv1d sft $DATASET]"
    python scripts/$script.py --mode conv1d --training-mode sft --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --native-dir $DATA/luna_native --interp-dir $DATA/interpolated --omneeg-dir $OMNEEG_DATA 2>&1 | tail -3
    echo ''
done

# BENDR Conv1d SFT
echo "[run_interpolate conv1d sft $DATASET]"
python scripts/run_interpolate_experiments.py --dataset $DATASET --use-conv1d-bridge --full-sft --start-seed 0 --n-seeds 1 --fast-dev-run --data-dir $DATA/interpolated 2>&1 | tail -3
echo ''

# BENDR OmnEEG SFT
echo "[run_omneeg sft $DATASET]"
python scripts/run_omneeg_experiments.py --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --data-dir $OMNEEG_DATA 2>&1 | tail -3

echo "=== $DATASET sweep complete at $(date) ==="
