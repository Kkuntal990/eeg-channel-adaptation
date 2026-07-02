#!/bin/bash
#SBATCH --job-name=ds_sweep
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=ddp483
#SBATCH --output=/expanse/projects/nemar/kuntal/adapter_finetuning_submission/logs/ds_sweep_%A_%a.out
#SBATCH --error=/expanse/projects/nemar/kuntal/adapter_finetuning_submission/logs/ds_sweep_%A_%a.err
#SBATCH --array=0-4

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
