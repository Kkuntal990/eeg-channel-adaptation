#!/usr/bin/env python
"""
Train BENDR on pre-computed OmnEEG 3D spherical harmonics features.

Standalone training script for conditions 7-9 of the alignment experiments.
Loads pre-computed OmnEEG features from HDF5, creates BENDR with encoder_only=True,
and trains with the same hyperparameters as existing adapter_finetuning experiments.

This script bypasses the Hydra/DataModule pipeline since the data is pre-computed
HDF5 rather than raw EEG loaded via MOABB/braindecode.

Usage:
    # Run single dataset with 15 seeds
    python scripts/run_omneeg_experiments.py --dataset bcic2a --n-seeds 15

    # Run all datasets
    python scripts/run_omneeg_experiments.py --dataset bcic2a physionet tuev --n-seeds 15

    # Quick test with 1 seed
    python scripts/run_omneeg_experiments.py --dataset bcic2a --n-seeds 1

    # Submit via SLURM (see submit section at bottom)
"""

import argparse
import logging
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, MetricCollection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Paths
DEFAULT_DATA_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/omneeg")
DEFAULT_OUTPUT_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/results/omneeg")

# Dataset configs matching existing experiments
DATASET_CONFIG = {
    "bcic2a": {
        "n_classes": 4,
        "batch_size": 64,
        "sfreq": 200.0,
        # Subject-based splits matching bcic2a.yaml
        "train_subjects": [1, 2, 3],
        "val_subjects": [4, 5, 6],
        "test_subjects": [7, 8, 9],
        "split_key": "subject",
    },
    "physionet": {
        "n_classes": 4,
        "batch_size": 32,
        "sfreq": 200.0,
        "train_subjects": list(range(1, 71)),
        "val_subjects": list(range(71, 90)),
        "test_subjects": list(range(90, 110)),
        "split_key": "subject",
    },
    "tuev": {
        "n_classes": 6,
        "batch_size": 32,
        "sfreq": 200.0,
        # TUEV uses split metadata (train/eval) rather than subject IDs
        # We'll handle this separately in the data loading
        "train_subjects": None,
        "val_subjects": None,
        "test_subjects": None,
        "split_key": "split",
    },
    "faced": {
        "n_classes": 9,
        "batch_size": 32,
        "sfreq": 200.0,
        "train_subjects": list(range(0, 80)),
        "val_subjects": list(range(80, 100)),
        "test_subjects": list(range(100, 123)),
        "split_key": "subject",
    },
    "isruc-sleep": {
        "n_classes": 5,
        "batch_size": 16,
        "sfreq": 200.0,
        "train_subjects": [
            "I011", "I013", "I014", "I017", "I019", "I020",
            "I021", "I023", "I025", "I027", "I028", "I029", "I030",
            "I031", "I032", "I033", "I034", "I035", "I036", "I037", "I038", "I039",
            "I041", "I042", "I043", "I044", "I045", "I046", "I047", "I048", "I049", "I050",
            "I051", "I052", "I053", "I054", "I055", "I056", "I057", "I058", "I059", "I060",
            "I061", "I062", "I063", "I064", "I065", "I066", "I067", "I068", "I069", "I070",
            "I071", "I072", "I073", "I074", "I075", "I076", "I077", "I078", "I079", "I080",
        ],
        "val_subjects": [
            "I081", "I082", "I083", "I084", "I085", "I086", "I087", "I088", "I089", "I090",
        ],
        "test_subjects": [
            "I091", "I092", "I093", "I094", "I095", "I096", "I097", "I098", "I099", "I100",
        ],
        "split_key": "subject",
    },
    "mdd_mumtaz2016": {
        "n_classes": 2,
        "batch_size": 32,
        "sfreq": 200.0,
        "train_subjects": [
            "HS1", "HS10", "HS11", "HS12", "HS13", "HS14", "HS15", "HS16", "HS17", "HS18", "HS19",
            "HS2", "HS20", "HS21", "HS22", "MDDS1", "MDDS10", "MDDS11", "MDDS12", "MDDS13", "MDDS14",
            "MDDS15", "MDDS16", "MDDS17", "MDDS18", "MDDS19", "MDDS2", "MDDS20", "MDDS21",
        ],
        "val_subjects": ["HS23", "HS24", "HS25", "MDDS22", "MDDS23", "MDDS24", "MDDS25"],
        "test_subjects": [
            "HS26", "HS27", "HS28", "HS29", "HS3", "HS30", "HS4", "HS5", "HS6", "HS7", "HS8", "HS9",
            "MDDS26", "MDDS27", "MDDS28", "MDDS29", "MDDS3", "MDDS30", "MDDS31", "MDDS32", "MDDS33",
            "MDDS34", "MDDS4", "MDDS5", "MDDS6", "MDDS7", "MDDS8", "MDDS9",
        ],
        "split_key": "subject",
    },
}

# Training hyperparameters (matching trainer/default.yaml)
TRAIN_CONFIG = {
    "max_epochs": 50,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "eta_min": 1e-6,
    "gradient_clip_val": 1.0,
    "patience": 15,
}

# BENDR pretrained config
BENDR_PRETRAINED_HUB = "braindecode/braindecode-bendr"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OmnEEGDataset(Dataset):
    """PyTorch Dataset for pre-computed OmnEEG features stored in HDF5."""

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.signals = f["signals"][:]  # [n_samples, n_coeffs, n_times]
            self.labels = f["labels"][:]    # [n_samples]
            self.subjects = f["subjects"][:]  # [n_samples]
            self.n_coeffs = f.attrs["n_coeffs"]
            self.n_times = f.attrs["n_times"]
            self.sfreq = f.attrs["sfreq"]

        # Apply min-max normalization per sample (matching BENDR preprocessing)
        self.signals = self._minmax_scale(self.signals)

        log.info(
            "Loaded %s: %d samples, shape %s",
            h5_path.name, len(self.signals), self.signals.shape,
        )

    @staticmethod
    def _minmax_scale(data):
        """Per-sample min-max scaling to [-1, 1], matching BENDR normalization."""
        # data shape: [n_samples, n_coeffs, n_times]
        mins = data.min(axis=(1, 2), keepdims=True)
        maxs = data.max(axis=(1, 2), keepdims=True)
        denom = maxs - mins
        denom[denom == 0] = 1.0  # Avoid division by zero
        return 2.0 * (data - mins) / denom - 1.0

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def split_dataset(dataset, config, seed=0):
    """Split dataset into train/val/test based on subject IDs or metadata."""
    subjects = dataset.subjects

    if config["split_key"] == "subject":
        # Decode bytes to strings if needed
        if len(subjects) > 0 and hasattr(subjects[0], "decode"):
            subjects = np.array([s.decode() for s in subjects])
        train_mask = np.isin(subjects, config["train_subjects"])
        val_mask = np.isin(subjects, config["val_subjects"])
        test_mask = np.isin(subjects, config["test_subjects"])
    elif config["split_key"] == "split":
        # TUEV: subject strings encode the split directory (e.g. paths under train/ or eval/).
        # Use this to reconstruct the official train/eval split matching tuev.yaml.
        if hasattr(subjects[0], "decode"):
            subjects_str = np.array([s.decode() for s in subjects])
        else:
            subjects_str = subjects.astype(str)

        # Samples from eval/ directory are the held-out test set
        # Samples from train/ directory are split 80/20 into train/val
        train_eval_mask = np.array(["train" in s.lower() for s in subjects_str])
        eval_mask = np.array(["eval" in s.lower() for s in subjects_str])

        # For any samples that don't match either pattern, put in train pool
        unmatched = ~train_eval_mask & ~eval_mask
        if unmatched.any():
            log.warning("%d samples matched neither train nor eval, adding to train pool", unmatched.sum())
            train_eval_mask = train_eval_mask | unmatched

        # Split the train pool into train/val (80/20) with a seeded RNG
        rng = np.random.default_rng(seed)
        train_pool_indices = np.where(train_eval_mask)[0]
        rng.shuffle(train_pool_indices)
        n_train = int(0.8 * len(train_pool_indices))

        n = len(dataset)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = eval_mask.copy()

        train_mask[train_pool_indices[:n_train]] = True
        val_mask[train_pool_indices[n_train:]] = True
    else:
        raise ValueError(f"Unknown split_key: {config['split_key']}")

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]

    log.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_indices), len(val_indices), len(test_indices),
    )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class BENDROmnEEGModule(pl.LightningModule):
    """Full BENDR (encoder + transformer) fine-tuned on OmnEEG features."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        eta_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        from braindecode.models import BENDR

        _BENDR_PRETRAINED_CHANS = 20

        # Channel projection: Conv1d(25, 20) maps OmnEEG SH coefficients
        # to pretrained channel space. All pretrained weights load (including layer 0).
        self.channel_proj = nn.Conv1d(n_chans, _BENDR_PRETRAINED_CHANS, kernel_size=1)
        log.info("Channel projection: Conv1d(%d, %d)", n_chans, _BENDR_PRETRAINED_CHANS)

        # Create BENDR with pretrained channel count (20)
        self.model = BENDR(
            n_chans=_BENDR_PRETRAINED_CHANS,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
            encoder_h=512,
            enc_width=(3, 2, 2, 2, 2, 2),
            enc_downsample=(3, 2, 2, 2, 2, 2),
            drop_prob=0.1,
            projection_head=False,
            start_token=-5,
            final_layer=True,
        )

        # Load pretrained weights (encoder layers 1-5 + full transformer load;
        # encoder layer 0 skipped due to shape mismatch 25 vs 20 input channels)
        self._load_pretrained()

        # Full SFT: all parameters trainable
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        log.info("Full SFT: %d/%d trainable (%.2f%%)", trainable, total, 100 * trainable / total)

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=n_outputs, average="macro"),
            "f1_macro": F1Score(task="multiclass", num_classes=n_outputs, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _load_pretrained(self):
        """Load pretrained BENDR encoder weights from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        log.info("Loading pretrained BENDR weights from %s", BENDR_PRETRAINED_HUB)

        try:
            try:
                path = hf_hub_download(
                    repo_id=BENDR_PRETRAINED_HUB,
                    filename="model.safetensors",
                )
                import safetensors.torch
                state_dict = safetensors.torch.load_file(path)
            except Exception:
                path = hf_hub_download(
                    repo_id=BENDR_PRETRAINED_HUB,
                    filename="pytorch_model.bin",
                )
                state_dict = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("Could not load pretrained weights: %s", e)
            return

        # Handle BENDR nested checkpoint structure
        # Checkpoint: {encoder_state_dict: {...}, contextualizer_state_dict: {...}}
        # Model expects: encoder.encoder.X, contextualizer.X
        if "encoder_state_dict" in state_dict:
            log.info("Detected BENDR nested checkpoint, flattening...")
            flat = {}
            for k, v in state_dict["encoder_state_dict"].items():
                flat["encoder." + k] = v
            for k, v in state_dict.get("contextualizer_state_dict", {}).items():
                flat["contextualizer." + k] = v
            state_dict = flat
            log.info("Flattened to %d keys", len(state_dict))

        # Filter by shape compatibility (strict=False)
        model_state = self.model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        log.info(
            "Loaded %d/%d pretrained weights (%d skipped, %d missing)",
            len(filtered), len(state_dict), len(skipped), len(missing),
        )

    def forward(self, x):
        # Channel projection: 25 SH coefficients -> 20 pretrained channels
        x = self.channel_proj(x)
        # Encoder-only mode: skip transformer, mean-pool encoder output
        encoded = self.model.encoder(x)  # [batch, 512, time/96]
        feature = encoded.mean(dim=2)    # [batch, 512]
        if self.model.final_layer is not None:
            feature = self.model.final_layer(feature)
        return feature

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        return {"loss": loss, "preds": preds, "targets": y}

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.train_metrics.update(out["preds"], out["targets"])
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return out["loss"]

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.val_metrics.update(out["preds"], out["targets"])
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.test_metrics.update(out["preds"], out["targets"])
        self.log("test_loss", out["loss"])

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        from adapter_finetuning.optim import CosineAnnealingWarmupLR

        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            max_epochs=self.hparams.max_epochs,
            eta_min=self.hparams.eta_min,
            warmup_start_lr=self.hparams.eta_min,
            warmup_epochs=self.hparams.warmup_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }


# ---------------------------------------------------------------------------
# Seed utilities (matching train.py)
# ---------------------------------------------------------------------------

def seed_everything_deterministic(seed: int):
    """Set seeds for reproducibility matching original implementation."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    pl.seed_everything(seed, workers=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_experiment(
    dataset_name: str,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    wandb_entity: str = "braindecode",
    wandb_project: str = "adapter-finetuning",
    fast_dev_run: bool = False,
):
    """Run a single OmnEEG experiment."""
    config = DATASET_CONFIG[dataset_name]
    h5_path = data_dir / f"{dataset_name}_omneeg_3d.h5"

    if not h5_path.exists():
        raise FileNotFoundError(
            f"OmnEEG features not found: {h5_path}\n"
            f"Run preprocess_omneeg.py first."
        )

    log.info("=" * 60)
    log.info("Dataset: %s, Seed: %d", dataset_name, seed)
    log.info("=" * 60)

    # Set seed
    seed_everything_deterministic(seed)

    # Load data
    dataset = OmnEEGDataset(h5_path)
    train_ds, val_ds, test_ds = split_dataset(dataset, config, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = BENDROmnEEGModule(
        n_chans=dataset.n_coeffs,  # 25 for resolution=4
        n_times=dataset.n_times,
        n_outputs=config["n_classes"],
        sfreq=dataset.sfreq,
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_epochs=TRAIN_CONFIG["warmup_epochs"],
        max_epochs=TRAIN_CONFIG["max_epochs"],
        eta_min=TRAIN_CONFIG["eta_min"],
    )

    # Wandb logger
    experiment_name = f"bendr_omneeg_sft_{dataset_name}_init{seed}"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    run_name = experiment_name
    if slurm_job_id:
        run_name = f"{run_name}_job{slurm_job_id}"

    tags = [
        "bendr", "omneeg", "full_sft", dataset_name,
        f"init{seed}", "alignment_experiment",
    ]
    if slurm_job_id:
        tags.append(f"slurm_{slurm_job_id}")

    wandb_logger = WandbLogger(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        group=f"bendr_omneeg_{dataset_name}",
        tags=tags,
        save_dir=str(output_dir),
        config={
            # Nested structure matching Hydra config schema so analyze_alignment.py
            # wandb filters (config.model.name, config.data.name, etc.) work correctly
            "model": {"name": "bendr", "encoder_only": True},
            "data": {"name": dataset_name, "n_classes": config["n_classes"], "batch_size": config["batch_size"]},
            "adapter": {"name": "full_sft"},
            "experiment": {"init_seed": seed},
            "preprocessing": "omneeg_3d",
            "n_chans": dataset.n_coeffs,
            "n_times": dataset.n_times,
            **TRAIN_CONFIG,
        },
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints" / (experiment_name + "_" + str(os.environ.get("SLURM_JOB_ID", "local")))),
            monitor="val_accuracy",
            mode="max",
            save_top_k=3,
            save_last=False,
            save_weights_only=True,
            filename="{epoch}-{val_accuracy:.4f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=TRAIN_CONFIG["patience"],
            mode="min",
            min_delta=0.001,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=TRAIN_CONFIG["max_epochs"],
        accelerator="auto",
        devices=1,
        precision=32,
        gradient_clip_val=TRAIN_CONFIG["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic="warn",
        fast_dev_run=fast_dev_run,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Get best val score
    best_score = 0.0
    if (
        isinstance(trainer.checkpoint_callback, ModelCheckpoint)
        and trainer.checkpoint_callback.best_model_score is not None
    ):
        score = trainer.checkpoint_callback.best_model_score
        best_score = score.item() if isinstance(score, torch.Tensor) else float(score)

    # Test
    if test_loader is not None and len(test_ds) > 0 and not fast_dev_run:
        trainer.test(model, test_loader, ckpt_path="best", weights_only=False)

    log.info("Best val accuracy: %.4f", best_score)
    return best_score


# ---------------------------------------------------------------------------
# SLURM submission helper
# ---------------------------------------------------------------------------

def submit_slurm_jobs(
    datasets: list[str],
    n_seeds: int,
    data_dir: Path,
    output_dir: Path,
    queue_size: int = 20,
):
    """Submit OmnEEG experiments as SLURM jobs via submitit."""
    import submitit

    PROJECT_ROOT = Path("/expanse/projects/nemar/kuntal/adapter_finetuning")
    PYTHON = "/expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning/bin/python"

    executor_config = {
        "slurm_partition": "gpu-shared",
        "slurm_account": "csd403",
        "slurm_qos": "gpu-shared-normal",
        "slurm_nodes": 1,
        "slurm_ntasks_per_node": 1,
        "slurm_cpus_per_task": 8,
        "mem_gb": 32,
        "timeout_min": 60,  # OmnEEG experiments are faster (~30 min)
        "slurm_setup": [
            "module load gpu",
            "module load cuda12.2/toolkit/12.2.2",
        ],
        "slurm_additional_parameters": {"gpus": 1},
        "slurm_srun_args": ["--export=ALL"],
    }

    jobs = []
    for dataset_name in datasets:
        for seed in range(n_seeds):
            experiment_name = f"bendr_omneeg_sft_{dataset_name}_init{seed}"

            executor = submitit.AutoExecutor(folder="submitit_logs")
            executor.update_parameters(name=experiment_name, **executor_config)

            job = executor.submit(
                run_experiment,
                dataset_name=dataset_name,
                seed=seed,
                data_dir=data_dir,
                output_dir=output_dir,
            )
            jobs.append((job, experiment_name))
            log.info("Submitted %s: job %s", experiment_name, job.job_id)

    log.info("Submitted %d jobs total", len(jobs))
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train BENDR on OmnEEG 3D spherical harmonics features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["bcic2a"],
        choices=["bcic2a", "physionet", "tuev", "faced", "isruc-sleep", "mdd_mumtaz2016"],
        help="Datasets to train on",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=15,
        help="Number of seeds to run per dataset (default: 15)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting seed index (default: 0). Runs seeds start_seed..start_seed+n_seeds-1",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing pre-computed OmnEEG HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results and checkpoints",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit as SLURM jobs instead of running locally",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Quick test with 1 batch (for debugging)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="braindecode",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="adapter-finetuning",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables for Expanse
    os.environ.setdefault("TMPDIR", "/expanse/projects/nemar/eeg_finetuning/.cache/tmp")
    os.environ.setdefault("HF_HOME", "/expanse/projects/nemar/eeg_finetuning/.cache/huggingface")
    os.environ.setdefault("WANDB_MODE", "online")

    if args.slurm:
        submit_slurm_jobs(
            datasets=args.dataset,
            n_seeds=args.n_seeds,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    else:
        results = {}
        for dataset_name in args.dataset:
            dataset_results = []
            for seed in range(args.start_seed, args.start_seed + args.n_seeds):
                score = run_experiment(
                    dataset_name=dataset_name,
                    seed=seed,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    wandb_entity=args.wandb_entity,
                    wandb_project=args.wandb_project,
                    fast_dev_run=args.fast_dev_run,
                )
                dataset_results.append(score)

            results[dataset_name] = dataset_results
            log.info(
                "%s: mean=%.4f, std=%.4f",
                dataset_name,
                np.mean(dataset_results),
                np.std(dataset_results),
            )

        # Print summary
        log.info("=" * 60)
        log.info("OmnEEG Experiment Results Summary")
        log.info("=" * 60)
        for dataset_name, scores in results.items():
            log.info(
                "  %s: %.4f +/- %.4f (n=%d)",
                dataset_name, np.mean(scores), np.std(scores), len(scores),
            )


if __name__ == "__main__":
    main()
