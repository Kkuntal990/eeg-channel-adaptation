# Channel Adaptation for EEG Foundation Models

Code for reproducing experiments in *"Matching EEG montages to foundation models: when learned projections, interpolation, and native handling each win"* (IEEE SMC 2026).

This repository implements a systematic benchmark of four channel adaptation methods across five pretrained EEG foundation models on five downstream tasks.

## Overview

- **Foundation models:** BENDR, Neuro-GPT, EEGPT, LUNA, CBraMod (5M--157M parameters)
- **Adaptation methods:** Native (model-internal), Conv1d projection, Spherical Spline Interpolation (SSI), OmnEEG spherical harmonics, Riemannian re-centering
- **Datasets:** BCIC2A, PhysioNet MI, TUEV, FACED, MDD Mumtaz
- **Training regimes:** Probe (frozen encoder) and SFT (supervised fine-tuning)

## Installation

```bash
conda create -y -n adapter-finetuning python=3.12 uv
conda activate adapter-finetuning
uv pip install -e .
```

## Repository Structure

```
adapter_finetuning/          # Minimal Python package (optim.py)
scripts/
  run_eegpt_experiments.py       # EEGPT training
  run_cbramod_experiments.py     # CBraMod training
  run_luna_experiments.py        # LUNA training
  run_neurogpt_experiments.py    # Neuro-GPT training
  run_interpolate_experiments.py # BENDR (Conv1d/SSI/Riemannian)
  run_omneeg_experiments.py      # BENDR OmnEEG (SFT only)
  preprocess_interpolate.py      # Precompute SSI / Riemannian HDF5
  preprocess_luna_native.py      # Precompute raw native HDF5
  preprocess_omneeg.py           # Precompute OmnEEG spherical harmonics HDF5
vendor/
  OmnEEG/                        # OmnEEG spherical-harmonic transform (vendored)
  NeuroGPT/                      # Neuro-GPT reference implementation
slurm/
  dummy_sanity_test.slurm        # Smoke test covering all scripts
configs/                         # (placeholder; scripts are self-contained)
```

## Pipeline

### 1. Preprocess (one-time per dataset)

```bash
# Spherical spline interpolation to 10--20 montage (needed for SSI + Riemannian + BENDR)
python scripts/preprocess_interpolate.py --output-dir $DATA_DIR/interpolated

# Raw native channels in HDF5 (needed for EEGPT/LUNA/CBraMod native mode)
python scripts/preprocess_luna_native.py --output-dir $DATA_DIR/luna_native

# OmnEEG spherical harmonic coefficients (25 channels, topology-agnostic)
python scripts/preprocess_omneeg.py --output-dir $DATA_DIR/omneeg --resolution 4
```

### 2. Run experiments

Each run script takes a common set of arguments:

- `--dataset {bcic2a, physionet, tuev, faced, mdd_mumtaz2016}`
- `--mode {conv1d, native, interpolated, omneeg, riemannian}` (EEGPT, CBraMod, Neuro-GPT)
- For **LUNA**: use `--mode {native, interpolated, omneeg, riemannian}` and add `--conv1d-bridge` on top of `--mode native` for the Conv1d column
- `--training-mode {probe, sft}` (probe = frozen encoder, sft = full fine-tune)
- `--start-seed INT --n-seeds INT` (we use 15 seeds per condition)
- `--fast-dev-run` (1-batch sanity check)

Examples:

```bash
# EEGPT, Conv1d adapter, SFT, 15 seeds on BCIC2A
python scripts/run_eegpt_experiments.py \
    --mode conv1d --training-mode sft --dataset bcic2a \
    --start-seed 0 --n-seeds 15

# Neuro-GPT, Riemannian alignment, probe, PhysioNet
python scripts/run_neurogpt_experiments.py \
    --mode riemannian --training-mode probe --dataset physionet \
    --start-seed 0 --n-seeds 15

# LUNA native + Conv1d bridge (this is the "Conv1d" column in the paper)
python scripts/run_luna_experiments.py \
    --mode native --conv1d-bridge --training-mode sft --dataset bcic2a \
    --start-seed 0 --n-seeds 15

# BENDR with Conv1d bridge, SFT
python scripts/run_interpolate_experiments.py \
    --dataset bcic2a --use-conv1d-bridge --full-sft \
    --start-seed 0 --n-seeds 15

# BENDR with SSI (default) probe
python scripts/run_interpolate_experiments.py \
    --dataset bcic2a --start-seed 0 --n-seeds 15

# BENDR with Riemannian recentering (use the riemannian HDF5 file)
python scripts/run_interpolate_experiments.py \
    --dataset bcic2a --full-sft \
    --h5-filename bcic2a_interpolated_spline_recenter_riemannian.h5 \
    --start-seed 0 --n-seeds 15

# BENDR OmnEEG SFT (probe not supported by script)
python scripts/run_omneeg_experiments.py \
    --dataset bcic2a --start-seed 0 --n-seeds 15
```

### 3. SLURM smoke test

```bash
sbatch slurm/dummy_sanity_test.slurm
```

Runs a `--fast-dev-run` pass for every script and every (method, training-mode) combination on BCIC2A. Verifies the pipeline is wired up before launching full experiments.

## Model Weights

Pretrained weights are loaded automatically via `huggingface_hub` and `braindecode` (`HF_HOME` must point to a writable cache). No manual download required.

## Citation

Please cite the associated paper if you use this code.
