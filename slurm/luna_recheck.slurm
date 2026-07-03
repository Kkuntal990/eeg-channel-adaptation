#!/bin/bash
#SBATCH --job-name=luna_check
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=CHANGE_ME        # <-- set your SLURM allocation
#SBATCH --output=logs/luna_check_%j.out
#SBATCH --error=logs/luna_check_%j.err

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

DATASET=bcic2a
echo '### LUNA smoke test with correct flags ###'

# LUNA native (no bridge)
for tmode in probe sft; do
    echo "[LUNA native $tmode $DATASET]"
    python scripts/run_luna_experiments.py --mode native --training-mode $tmode --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --native-dir $DATA/luna_native --interp-dir $DATA/interpolated --omneeg-dir $OMNEEG_DATA 2>&1 | tail -3
done

# LUNA native + conv1d bridge (this is the 'Conv1d' column in the paper)
for tmode in probe sft; do
    echo "[LUNA native+conv1d-bridge $tmode $DATASET]"
    python scripts/run_luna_experiments.py --mode native --conv1d-bridge --training-mode $tmode --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --native-dir $DATA/luna_native --interp-dir $DATA/interpolated --omneeg-dir $OMNEEG_DATA 2>&1 | tail -3
done

# LUNA interpolated, omneeg, riemannian
for mode in interpolated omneeg riemannian; do
    for tmode in probe sft; do
        echo "[LUNA $mode $tmode $DATASET]"
        python scripts/run_luna_experiments.py --mode $mode --training-mode $tmode --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --native-dir $DATA/luna_native --interp-dir $DATA/interpolated --omneeg-dir $OMNEEG_DATA 2>&1 | tail -3
    done
done
echo '=== LUNA test complete ==='
