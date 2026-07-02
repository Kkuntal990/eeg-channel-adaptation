"""Core module for adapter-based EEG foundation model fine-tuning.

This package provides:
- adapters: PEFT adapter implementations (LoRA, IA3, AdaLoRA, etc.)
- callbacks: PyTorch Lightning callback utilities
- datamodule: PyTorch Lightning DataModule for EEG data
- lightning_module: PyTorch Lightning Module for training
- config_schemas: Pydantic configuration classes
"""

import torch
if torch.cuda.is_initialized():
    print("DEBUG: CUDA already initialized when entering adapter_finetuning package!")

from adapter_finetuning.adapters import (
    apply_peft_to_model,
    PeftModelWrapper,
    AdapterConfig,
    LoraConfig,
    IA3Config,
    AdaLoraConfig,
    FullFtConfig,
    DoraConfig,
    OFTConfig,
    MODEL_TARGET_MODULES,
    MODEL_FF_MODULES,
)
from adapter_finetuning.callbacks import (
    get_best_validation_score,
    run_test_phase,
)
from adapter_finetuning.config_schemas import (
    # Utils
    InstantiatorConfig,
    PathInstantiatorConfig,
    instantiate_optional_list,
    # Loaders
    BaseLoaderConfig,
    MOABBLoaderConfig,
    EEGDashLoaderConfig,
    LoaderConfig,
    # Preprocessing
    EEGPrepConfig,
    StandardizeConfig,
    PreprocessorConfig,
    # Windowers
    BaseWindowerConfig,
    EventsWindowerConfig,
    FixedLengthWindowerConfig,
    WindowerConfig,
    # Splitters
    BaseSplitterConfig,
    RandomSplitterConfig,
    CrossSubjectSplitterConfig,
    CrossSessionSplitterConfig,
    SplitterConfig,
)

__all__ = [
    # Adapters
    "apply_peft_to_model",
    "PeftModelWrapper",
    "AdapterConfig",
    "LoraConfig",
    "IA3Config",
    "AdaLoraConfig",
    "FullFtConfig",
    "DoraConfig",
    "OFTConfig",
    "MODEL_TARGET_MODULES",
    "MODEL_FF_MODULES",
    # Callbacks
    "get_best_validation_score",
    "run_test_phase",
    # Config schemas
    "InstantiatorConfig",
    "PathInstantiatorConfig",
    "instantiate_optional_list",
    "BaseLoaderConfig",
    "MOABBLoaderConfig",
    "EEGDashLoaderConfig",
    "LoaderConfig",
    "EEGPrepConfig",
    "StandardizeConfig",
    "PreprocessorConfig",
    "BaseWindowerConfig",
    "EventsWindowerConfig",
    "FixedLengthWindowerConfig",
    "WindowerConfig",
    "BaseSplitterConfig",
    "RandomSplitterConfig",
    "CrossSubjectSplitterConfig",
    "CrossSessionSplitterConfig",
    "SplitterConfig",
]
