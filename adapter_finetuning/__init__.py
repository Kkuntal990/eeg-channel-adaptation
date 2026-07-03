"""Minimal support package for the EEG channel-adaptation experiments.

The training/preprocessing scripts in ``scripts/`` are self-contained; the only
shared utility they import is the LR scheduler in :mod:`adapter_finetuning.optim`.
"""

from adapter_finetuning.optim import CosineAnnealingWarmupLR

__all__ = ["CosineAnnealingWarmupLR"]
