#!/bin/bash
#SBATCH --job-name=luna_check
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=ddp483
#SBATCH --output=/expanse/projects/nemar/kuntal/adapter_finetuning_submission/logs/luna_check_%j.out
#SBATCH --error=/expanse/projects/nemar/kuntal/adapter_finetuning_submission/logs/luna_check_%j.err

REPO=/expanse/projects/nemar/kuntal/adapter_finetuning_submission
DATA=/expanse/projects/nemar/kuntal/adapter_finetuning/data
OMNEEG_DATA=/expanse/projects/nemar/adapter_finetuning/data/omneeg

cd $REPO
mkdir -p logs

module purge
module load gpu
module load cuda12.2/toolkit/12.2.2
module load anaconda3/2021.05
source activate /expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning

export HF_HOME="/expanse/projects/nemar/dtyoung/huggingface_cache"
export MNE_DATA="/expanse/projects/nemar/dtyoung/mne_data"
export TMPDIR="/expanse/projects/nemar/eeg_finetuning/.cache/tmp"
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
