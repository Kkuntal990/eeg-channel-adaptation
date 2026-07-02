#!/bin/bash
#SBATCH --job-name=sanity_test
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=ddp483
#SBATCH --output=/expanse/projects/nemar/kuntal/adapter_finetuning_submission/logs/sanity_%A_%a.out
#SBATCH --error=/expanse/projects/nemar/kuntal/adapter_finetuning_submission/logs/sanity_%A_%a.err
#SBATCH --array=0-5

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
echo "=== Task $SLURM_ARRAY_TASK_ID starting at $(date) ==="

case $SLURM_ARRAY_TASK_ID in
    0)
        echo '### EEGPT smoke test ###'
        for mode in conv1d native interpolated omneeg riemannian; do
            for tmode in probe sft; do
                echo "[EEGPT $mode $tmode $DATASET]"
                python scripts/run_eegpt_experiments.py --mode $mode --training-mode $tmode --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --native-dir $DATA/luna_native --interp-dir $DATA/interpolated --data-dir $OMNEEG_DATA 2>&1 | tail -3
                echo ''
            done
        done
        ;;
    1)
        echo '### CBraMod smoke test ###'
        for mode in conv1d native interpolated omneeg riemannian; do
            for tmode in probe sft; do
                echo "[CBraMod $mode $tmode $DATASET]"
                python scripts/run_cbramod_experiments.py --mode $mode --training-mode $tmode --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --native-dir $DATA/luna_native --interp-dir $DATA/interpolated --data-dir $OMNEEG_DATA 2>&1 | tail -3
                echo ''
            done
        done
        ;;
    2)
        echo '### LUNA smoke test ###'
        for mode in conv1d native interpolated omneeg riemannian; do
            for tmode in probe sft; do
                echo "[LUNA $mode $tmode $DATASET]"
                python scripts/run_luna_experiments.py --mode $mode --training-mode $tmode --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --native-dir $DATA/luna_native --interp-dir $DATA/interpolated --data-dir $OMNEEG_DATA 2>&1 | tail -3
                echo ''
            done
        done
        ;;
    3)
        echo '### Neuro-GPT smoke test ###'
        for mode in conv1d interpolated omneeg riemannian; do
            for tmode in probe sft; do
                echo "[Neuro-GPT $mode $tmode $DATASET]"
                python scripts/run_neurogpt_experiments.py --mode $mode --training-mode $tmode --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --interp-dir $DATA/interpolated --data-dir $OMNEEG_DATA 2>&1 | tail -3
                echo ''
            done
        done
        ;;
    4)
        echo '### BENDR (interpolate script) smoke test ###'
        # SSI probe/sft
        for sft_flag in '' '--full-sft'; do
            echo "[BENDR SSI $sft_flag $DATASET]"
            python scripts/run_interpolate_experiments.py --dataset $DATASET $sft_flag --start-seed 0 --n-seeds 1 --fast-dev-run --data-dir $DATA/interpolated 2>&1 | tail -3
            echo ''
        done
        # Conv1d probe/sft
        for sft_flag in '' '--full-sft'; do
            echo "[BENDR Conv1d $sft_flag $DATASET]"
            python scripts/run_interpolate_experiments.py --dataset $DATASET --use-conv1d-bridge $sft_flag --start-seed 0 --n-seeds 1 --fast-dev-run --data-dir $DATA/interpolated 2>&1 | tail -3
            echo ''
        done
        # Riemannian probe/sft (different h5 file)
        for sft_flag in '' '--full-sft'; do
            echo "[BENDR Riemannian $sft_flag $DATASET]"
            python scripts/run_interpolate_experiments.py --dataset $DATASET $sft_flag --h5-filename ${DATASET}_interpolated_spline_recenter_riemannian.h5 --start-seed 0 --n-seeds 1 --fast-dev-run --data-dir $DATA/interpolated 2>&1 | tail -3
            echo ''
        done
        ;;
    5)
        echo '### BENDR OmnEEG smoke test ###'
        echo "[BENDR OmnEEG sft $DATASET]"
        python scripts/run_omneeg_experiments.py --dataset $DATASET --start-seed 0 --n-seeds 1 --fast-dev-run --data-dir $OMNEEG_DATA 2>&1 | tail -3
        ;;
esac

echo "=== Task $SLURM_ARRAY_TASK_ID complete at $(date) ==="
