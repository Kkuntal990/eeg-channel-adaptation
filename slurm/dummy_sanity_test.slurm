#!/bin/bash
#SBATCH --job-name=sanity_test
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=CHANGE_ME        # <-- set your SLURM allocation
#SBATCH --output=logs/sanity_%A_%a.out
#SBATCH --error=logs/sanity_%A_%a.err
#SBATCH --array=0-5

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
