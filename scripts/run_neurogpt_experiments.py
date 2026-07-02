#!/usr/bin/env python
"""
Train Neuro-GPT on EEG datasets with multiple input modes and training regimes.

Neuro-GPT (Cui et al., IEEE ISBI 2024) is an EEG foundation model with:
  - EEG encoder: EEGConformer (2 conv layers + 6 self-attention layers, n_filters_time=40)
  - Embedder: Linear projection from encoder output (1080) to GPT embed_dim (1024)
  - GPT-2 decoder: 4 layers, 16 heads, embed_dim=1024
  - ~79.5M params
  - FIXED 22 channels (extended 10-20): Fp1 Fp2 F7 F3 Fz F4 F8 T1 T3 C3 Cz C4 T4 T2 T5 P3 Pz P4 T6 O1 Oz O2
  - Pretrained on TUH EEG Corpus at 250 Hz
  - Pretrained weights: HuggingFace wenhuic/Neuro-GPT

Architecture expects input shape (batch, num_chunks, 22, chunk_len) where chunk_len=500 at 250Hz.
The encoder produces features of shape (batch*chunks, seq_len, 40) which are flattened to
parcellation_dim = ((chunk_len - 25 + 1 - 75) // 15 + 1) * 40 = 1080 for chunk_len=500.

Modes:
  conv1d:       Raw EEG (native channels) -> Conv1d bridge -> 22ch -> Neuro-GPT.
  interpolated: 19ch SSI data -> Conv1d(19,22) bridge -> 22ch -> Neuro-GPT.
  omneeg:       25 SH coefficients -> Conv1d(25,22) bridge -> 22ch -> Neuro-GPT.
  riemannian:   19ch Riemannian re-centered -> Conv1d(19,22) bridge -> 22ch -> Neuro-GPT.

All modes require a Conv1d bridge since our data never has exactly the 22 Neuro-GPT channels.

Training:
  probe: Freeze encoder+embedder+decoder, train only bridge + decoding_head. lr=5e-4
  sft:   Unfreeze all parameters. lr=1e-5

Usage:
    python scripts/run_neurogpt_experiments.py --mode interpolated --training-mode probe --dataset bcic2a
    python scripts/run_neurogpt_experiments.py --mode conv1d --training-mode sft --dataset bcic2a physionet tuev
"""

import argparse
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, MetricCollection

# For building Neuro-GPT from vendor code
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import GPT2Config, GPT2Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Paths
DEFAULT_NATIVE_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/luna_native_raw")
DEFAULT_INTERP_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/interpolated_raw")
DEFAULT_OMNEEG_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/omneeg")
DEFAULT_OUTPUT_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/results/neurogpt")

NEUROGPT_SFREQ = 250.0       # Neuro-GPT pretrained at 250 Hz
NEUROGPT_N_CHANS = 22        # Fixed 22 channels
NEUROGPT_CHUNK_LEN = 500     # 2 seconds at 250 Hz
NEUROGPT_CHUNK_OVLP = 50     # 0.2 second overlap
NEUROGPT_N_FILTERS_TIME = 40
NEUROGPT_FILTER_TIME_LEN = 25
NEUROGPT_POOL_TIME_LEN = 75
NEUROGPT_STRIDE_AVG_POOL = 15
NEUROGPT_ATT_DEPTH = 6
NEUROGPT_ATT_HEADS = 10
NEUROGPT_ATT_DROP = 0.5
NEUROGPT_DROP_PROB = 0.5

# Encoder output dimension: ((500-25+1-75)//15+1)*40 = 27*40 = 1080
NEUROGPT_PARCELLATION_DIM = (
    (NEUROGPT_CHUNK_LEN - NEUROGPT_FILTER_TIME_LEN + 1 - NEUROGPT_POOL_TIME_LEN)
    // NEUROGPT_STRIDE_AVG_POOL + 1
) * NEUROGPT_N_FILTERS_TIME  # = 1080

# GPT config
NEUROGPT_EMBED_DIM = 1024
NEUROGPT_GPT_LAYERS = 6
NEUROGPT_GPT_HEADS = 16
NEUROGPT_N_POSITIONS = 512
NEUROGPT_DROPOUT = 0.1

NEUROGPT_HF_REPO = "wenhuic/Neuro-GPT"
NEUROGPT_HF_FILE = "pretrained_model/pytorch_model.bin"

# Neuro-GPT's expected 22 channel order
NEUROGPT_22_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
    "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2",
]

DATASET_CONFIG = {
    "bcic2a": {
        "n_classes": 4,
        "batch_size": 64,
        "train_subjects": [1, 2, 3],
        "val_subjects": [4, 5, 6],
        "test_subjects": [7, 8, 9],
        "split_key": "subject",
    },
    "physionet": {
        "n_classes": 4,
        "batch_size": 32,
        "train_subjects": list(range(1, 71)),
        "val_subjects": list(range(71, 90)),
        "test_subjects": list(range(90, 110)),
        "split_key": "subject",
    },
    "tuev": {
        "n_classes": 6,
        "batch_size": 32,
        "train_subjects": None,
        "val_subjects": None,
        "test_subjects": None,
        "split_key": "split",
    },
    "faced": {
        "n_classes": 9,
        "batch_size": 32,
        "train_subjects": list(range(0, 80)),
        "val_subjects": list(range(80, 100)),
        "test_subjects": list(range(100, 123)),
        "split_key": "subject",
    },
    "mdd_mumtaz2016": {
        "n_classes": 2,
        "batch_size": 32,
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

TRAIN_CONFIG_PROBE = {
    "max_epochs": 50,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "eta_min": 1e-6,
    "gradient_clip_val": 1.0,
    "patience": 10,
}

TRAIN_CONFIG_SFT = {
    "max_epochs": 50,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "eta_min": 1e-6,
    "gradient_clip_val": 1.0,
    "patience": 10,
}


# ---------------------------------------------------------------------------
# Neuro-GPT Model Components (reconstructed from vendor code)
# ---------------------------------------------------------------------------

class _PatchEmbedding(nn.Module):
    """Patch embedding from EEGConformer: temporal conv + spatial conv + pool."""
    def __init__(self, n_filters_time, filter_time_length, n_channels,
                 pool_time_length, stride_avg_pool, drop_prob):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)),
            nn.Dropout(p=drop_prob),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, att_heads, att_drop, forward_expansion=4):
        super().__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _MultiHeadAttention(emb_size, att_heads, att_drop),
                nn.Dropout(att_drop),
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=att_drop),
                nn.Dropout(att_drop),
            )),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__(
            *[_TransformerEncoderBlock(emb_size, att_heads, att_drop) for _ in range(att_depth)]
        )


class EEGConformerEncoder(nn.Module):
    """EEG Conformer encoder (no classification head).
    Input: (batch*chunks, 1, n_chans, chunk_len)
    Output: (batch*chunks, seq_len, n_filters_time)
    """
    def __init__(self, n_chans=22, n_filters_time=40, filter_time_length=25,
                 pool_time_length=75, pool_time_stride=15, drop_prob=0.5,
                 att_depth=6, att_heads=10, att_drop_prob=0.5):
        super().__init__()
        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
        )
        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob,
        )

    def forward(self, x):
        """x: (batch*chunks, n_chans, chunk_len)"""
        batch_chunks, chann, time = x.size()
        x = x.unsqueeze(1)  # (batch*chunks, 1, chans, time)
        x = self.patch_embedding(x)  # (batch*chunks, seq_len, n_filters_time)
        x = self.transformer(x)
        return x


class EmbeddingModel(nn.Module):
    """Projects from parcellation_dim to embed_dim."""
    def __init__(self, in_dim=1080, embed_dim=1024, num_hidden_layers=1, dropout=0.1):
        super().__init__()
        layer_stack = []
        for _ in range(num_hidden_layers - 1):
            layer_stack.extend([
                nn.Linear(in_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ])
        layer_stack.extend([
            nn.Linear(embed_dim if num_hidden_layers > 1 else in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
        ])
        self.model = nn.Sequential(*layer_stack)

    def forward(self, inputs):
        # inputs: (batch, seq, in_dim) -> (batch, seq, embed_dim)
        b, s, d = inputs.size()
        x = inputs.reshape(b * s, d)
        x = self.model(x)
        return x.reshape(b, s, -1)


class CSMEmbedder(nn.Module):
    """Embedder that projects encoder features to GPT embedding space.
    Also manages CLS/MSK tokens for the CSM training paradigm.
    For our fine-tuning, we mainly need the embedding projection + CLS token."""
    def __init__(self, in_dim=1080, embed_dim=1024, num_hidden_layers=1, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.embed_model = EmbeddingModel(in_dim, embed_dim, num_hidden_layers, dropout)
        self.msk_embed = nn.Parameter(torch.empty(1, 1, in_dim))
        self.cls_embed = nn.Parameter(torch.empty(1, 1, in_dim))
        nn.init.normal_(self.msk_embed, mean=0.0, std=1.0)
        nn.init.normal_(self.cls_embed, mean=0.0, std=1.0)
        self.is_decoding_mode = False

    def switch_decoding_mode(self, is_decoding_mode=False):
        self.is_decoding_mode = is_decoding_mode
        self.training_style = 'decoding' if is_decoding_mode else 'CSM'

    def add_cls_embed(self, batch):
        """Add CLS token at the end of each valid sequence."""
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        batch_size = batch[inputs_key].size(0)
        sequence_lengths = batch['attention_mask'].sum(dim=1)
        inputs_embeds = []
        for i in range(batch_size):
            sl = int(sequence_lengths[i].item())
            inputs_embeds.append(torch.cat([
                batch[inputs_key][i, :sl, :],
                self.cls_embed[0],
                batch[inputs_key][i, sl:, :],
            ], dim=0))
        batch['inputs_embeds'] = torch.stack(inputs_embeds, dim=0)
        # Update attention mask to account for CLS token
        if 'attention_mask' in batch:
            filling = torch.ones(batch_size, 1, device=batch['attention_mask'].device, dtype=torch.long)
            batch['attention_mask'] = torch.cat([filling, batch['attention_mask']], dim=1)
        return batch

    def forward(self, batch):
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        if self.in_dim == self.embed_dim:
            return batch[inputs_key]
        return self.embed_model(batch[inputs_key])


class GPTDecoder(nn.Module):
    """GPT-2 decoder with pooler and classification head for decoding mode."""
    def __init__(self, embed_dim=1024, num_hidden_layers=4, num_attention_heads=16,
                 n_positions=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.config = GPT2Config(
            vocab_size=1,
            n_positions=n_positions,
            n_embd=embed_dim,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            n_inner=embed_dim * 4,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            embd_pdrop=dropout,
            activation_function='gelu_new',
        )
        self.transformer = GPT2Model(config=self.config)
        self.is_decoding_mode = False
        self.pooler_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.decoding_head = None
        self.num_decoding_classes = None

    def switch_decoding_mode(self, is_decoding_mode=False, num_decoding_classes=None):
        self.is_decoding_mode = is_decoding_mode
        if is_decoding_mode and num_decoding_classes is not None:
            self.num_decoding_classes = num_decoding_classes
            self.decoding_head = nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(32, num_decoding_classes),
            )

    def forward(self, batch):
        transformer_outputs = self.transformer.forward(
            inputs_embeds=batch['inputs_embeds'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
        )
        outputs = {'outputs': transformer_outputs['last_hidden_state']}
        if self.is_decoding_mode:
            hidden = outputs['outputs']
            batch_size = hidden.size(0)
            sequence_lengths = batch['attention_mask'].sum(dim=1) - 1
            pooled = self.pooler_layer(
                hidden[torch.arange(batch_size, device=hidden.device), sequence_lengths]
            )
            outputs['decoding_logits'] = self.decoding_head(pooled)
        return outputs


class UnEmbedder(nn.Module):
    """Projects from embed_dim back to parcellation_dim (for pretraining loss)."""
    def __init__(self, embed_dim=1024, out_dim=1080, num_hidden_layers=1, dropout=0.1):
        super().__init__()
        layer_stack = []
        for _ in range(num_hidden_layers - 1):
            layer_stack.extend([
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ])
        layer_stack.append(nn.Linear(embed_dim, out_dim))
        self.model = nn.Sequential(*layer_stack)

    def forward(self, inputs):
        b, s, d = inputs.size()
        x = inputs.reshape(b * s, d)
        x = self.model(x)
        return {'outputs': x.reshape(b, s, -1)}


class NeuroGPTModel(nn.Module):
    """
    Full Neuro-GPT model: encoder -> embedder -> GPT decoder.
    Reconstructed from vendor code to avoid import path issues.
    """
    def __init__(self, n_classes, num_chunks=4):
        super().__init__()
        self.num_chunks = num_chunks
        self.n_classes = n_classes

        # EEG Conformer encoder (fixed 22 channels)
        self.encoder = EEGConformerEncoder(
            n_chans=NEUROGPT_N_CHANS,
            n_filters_time=NEUROGPT_N_FILTERS_TIME,
            filter_time_length=NEUROGPT_FILTER_TIME_LEN,
            pool_time_length=NEUROGPT_POOL_TIME_LEN,
            pool_time_stride=NEUROGPT_STRIDE_AVG_POOL,
            drop_prob=NEUROGPT_DROP_PROB,
            att_depth=NEUROGPT_ATT_DEPTH,
            att_heads=NEUROGPT_ATT_HEADS,
            att_drop_prob=NEUROGPT_ATT_DROP,
        )

        # Embedder: projects encoder output to GPT space
        self.embedder = CSMEmbedder(
            in_dim=NEUROGPT_PARCELLATION_DIM,  # 1080
            embed_dim=NEUROGPT_EMBED_DIM,      # 1024
            num_hidden_layers=1,
            dropout=NEUROGPT_DROPOUT,
        )

        # GPT-2 decoder
        self.decoder = GPTDecoder(
            embed_dim=NEUROGPT_EMBED_DIM,
            num_hidden_layers=NEUROGPT_GPT_LAYERS,
            num_attention_heads=NEUROGPT_GPT_HEADS,
            n_positions=NEUROGPT_N_POSITIONS,
            dropout=NEUROGPT_DROPOUT,
        )

        # UnEmbedder (needed for loading pretrained weights)
        self.unembedder = UnEmbedder(
            embed_dim=NEUROGPT_EMBED_DIM,
            out_dim=NEUROGPT_PARCELLATION_DIM,
            num_hidden_layers=1,
            dropout=NEUROGPT_DROPOUT,
        )

        # Switch to decoding (classification) mode
        self.embedder.switch_decoding_mode(is_decoding_mode=True)
        self.decoder.switch_decoding_mode(
            is_decoding_mode=True,
            num_decoding_classes=n_classes,
        )

    def forward(self, inputs, attention_mask):
        """
        Args:
            inputs: (batch, num_chunks, 22, chunk_len)
            attention_mask: (batch, num_chunks) - 1 for valid chunks, 0 for padding
        Returns:
            logits: (batch, n_classes)
        """
        batch_size, nchunks, chann, time = inputs.size()

        # Encoder: process each chunk independently
        x = inputs.reshape(batch_size * nchunks, chann, time)
        features = self.encoder(x)  # (batch*chunks, seq_len, n_filters_time)
        b_c, f1, f2 = features.size()
        # Flatten seq_len * n_filters_time -> parcellation_dim
        features = features.reshape(batch_size, nchunks, f1 * f2)

        # Build batch dict for embedder/decoder
        batch = {
            'inputs': features,
            'attention_mask': attention_mask,
        }

        # Add CLS token
        batch = self.embedder.add_cls_embed(batch)

        # Embed
        batch['inputs_embeds'] = self.embedder(batch)

        # GPT decode
        outputs = self.decoder(batch)

        return outputs['decoding_logits']


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeuroGPTDataset(Dataset):
    """Dataset that loads HDF5 and chunks data for Neuro-GPT."""

    def __init__(self, h5_path: Path, mode: str = "interpolated",
                 chunk_len: int = NEUROGPT_CHUNK_LEN,
                 chunk_ovlp: int = NEUROGPT_CHUNK_OVLP,
                 max_chunks: int = 4,
                 target_sfreq: float = NEUROGPT_SFREQ):
        self.mode = mode
        self.chunk_len = chunk_len
        self.chunk_ovlp = chunk_ovlp
        self.max_chunks = max_chunks
        self.target_sfreq = target_sfreq

        with h5py.File(h5_path, "r") as f:
            self.signals = f["signals"][:]
            self.labels = f["labels"][:]
            self.subjects = f["subjects"][:]
            self.n_channels = int(f.attrs["n_channels"]) if "n_channels" in f.attrs else self.signals.shape[1]
            self.n_times = int(f.attrs["n_times"]) if "n_times" in f.attrs else self.signals.shape[2]
            self.sfreq = float(f.attrs.get("sfreq", 200.0))

            if "channel_names" in f:
                raw_names = f["channel_names"][:]
                self.channel_names = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in raw_names
                ]
            else:
                self.channel_names = None

        # Resample to target sfreq if needed
        if abs(self.sfreq - target_sfreq) > 1.0:
            log.info("Resampling from %.0fHz to %.0fHz", self.sfreq, target_sfreq)
            import scipy.signal as sig
            new_n_times = int(self.n_times * target_sfreq / self.sfreq)
            self.signals = sig.resample(self.signals, new_n_times, axis=2).astype(np.float32)
            self.n_times = new_n_times
            self.sfreq = target_sfreq

        log.info(
            "Loaded %s: %d samples, shape (%d, %d, %d), sfreq=%.0f, mode=%s, max_chunks=%d",
            h5_path.name, len(self.signals), len(self.signals),
            self.n_channels, self.n_times, self.sfreq, mode, max_chunks,
        )

    def __len__(self):
        return len(self.signals)

    def _normalize(self, x):
        """Per-sample, per-channel z-normalization (matching Neuro-GPT pretraining)."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + 1e-25)

    def _chunk_signal(self, signal):
        """Split a (C, T) signal into chunks of (num_chunks, C, chunk_len).
        Returns chunks and attention_mask."""
        n_chans, total_len = signal.shape
        chunks = []
        start = 0
        while len(chunks) < self.max_chunks and start + self.chunk_len <= total_len:
            chunks.append(signal[:, start:start + self.chunk_len])
            start += self.chunk_len - self.chunk_ovlp

        actual_chunks = len(chunks)
        # Pad to max_chunks if needed
        while len(chunks) < self.max_chunks:
            chunks.append(np.zeros((n_chans, self.chunk_len), dtype=np.float32))

        attention_mask = np.zeros(self.max_chunks, dtype=np.float32)
        attention_mask[:actual_chunks] = 1.0

        return np.stack(chunks, axis=0), attention_mask  # (max_chunks, C, chunk_len), (max_chunks,)

    def __getitem__(self, idx):
        x = self.signals[idx].copy()  # (C, T)

        if self.mode == "omneeg":
            # OmneEG: SH coefficients, apply mild normalization
            x = x / 1e4
        else:
            # Raw EEG in Volts: convert to uV and z-normalize per channel
            x = x * 1e6
            x = self._normalize(x)

        chunks, attn_mask = self._chunk_signal(x)
        chunks = torch.tensor(chunks, dtype=torch.float32)      # (max_chunks, C, chunk_len)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)    # (max_chunks,)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return chunks, attn_mask, y


def split_dataset(dataset, config, seed=0):
    """Split dataset into train/val/test based on subject IDs or metadata."""
    subjects = dataset.subjects

    if config["split_key"] == "subject":
        if len(subjects) > 0 and hasattr(subjects[0], "decode"):
            subjects = np.array([s.decode() for s in subjects])
        train_mask = np.isin(subjects, config["train_subjects"])
        val_mask = np.isin(subjects, config["val_subjects"])
        test_mask = np.isin(subjects, config["test_subjects"])
    elif config["split_key"] == "split":
        if hasattr(subjects[0], "decode"):
            subjects_str = np.array([s.decode() for s in subjects])
        else:
            subjects_str = subjects.astype(str)

        train_eval_mask = np.array(["train" in s.lower() for s in subjects_str])
        eval_mask = np.array(["eval" in s.lower() for s in subjects_str])
        unmatched = ~train_eval_mask & ~eval_mask
        if unmatched.any():
            log.warning("%d samples matched neither train nor eval, adding to train pool", unmatched.sum())
            train_eval_mask = train_eval_mask | unmatched

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

    log.info(
        "Split: train=%d, val=%d, test=%d",
        train_mask.sum(), val_mask.sum(), test_mask.sum(),
    )
    return (
        Subset(dataset, np.where(train_mask)[0]),
        Subset(dataset, np.where(val_mask)[0]),
        Subset(dataset, np.where(test_mask)[0]),
    )


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class NeuroGPTExperimentModule(pl.LightningModule):
    """Neuro-GPT model with Conv1d bridge and configurable training mode."""

    def __init__(
        self,
        n_chans_input: int,
        n_outputs: int,
        num_chunks: int = 4,
        use_bridge: bool = True,
        training_mode: str = "probe",
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        eta_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_bridge = use_bridge

        # Conv1d bridge: project input channels -> 22 channels
        self.bridge = None
        if use_bridge and n_chans_input != NEUROGPT_N_CHANS:
            self.bridge = nn.Conv1d(n_chans_input, NEUROGPT_N_CHANS, kernel_size=1)
            # Initialize with identity-like mapping where possible
            nn.init.eye_(self.bridge.weight[:, :min(n_chans_input, NEUROGPT_N_CHANS), 0])
            nn.init.zeros_(self.bridge.bias)
            log.info("Conv1d bridge: %d -> %d channels", n_chans_input, NEUROGPT_N_CHANS)
        elif n_chans_input == NEUROGPT_N_CHANS:
            log.info("No bridge needed: input already has %d channels", NEUROGPT_N_CHANS)

        # Build Neuro-GPT model
        self.model = NeuroGPTModel(n_classes=n_outputs, num_chunks=num_chunks)

        # Load pretrained weights
        self._load_pretrained()

        # Freeze based on training mode
        if training_mode == "probe":
            for name, param in self.model.named_parameters():
                # Freeze everything except the decoding head
                if 'decoding_head' not in name:
                    param.requires_grad = False
            # Bridge is always trainable (if present)
            if self.bridge is not None:
                for param in self.bridge.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(
            "Training mode: %s | Trainable: %d/%d (%.2f%%)",
            training_mode, trainable, total, 100 * trainable / total,
        )

        self.criterion = nn.CrossEntropyLoss()
        metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=n_outputs, average="macro"),
            "f1_macro": F1Score(task="multiclass", num_classes=n_outputs, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _load_pretrained(self):
        """Load pretrained Neuro-GPT weights from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            log.warning("huggingface_hub not available; skipping pretrained weight loading.")
            return

        log.info("Loading pretrained Neuro-GPT weights from %s", NEUROGPT_HF_REPO)
        try:
            path = hf_hub_download(repo_id=NEUROGPT_HF_REPO, filename=NEUROGPT_HF_FILE)
            pretrained = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("Could not download pretrained weights: %s", e)
            return

        # Map vendor state dict keys to our reconstructed model keys
        key_map = self._build_key_mapping(pretrained)
        model_state = self.model.state_dict()
        compatible = {}
        skipped = []

        for src_key, tgt_key in key_map.items():
            if tgt_key in model_state:
                if pretrained[src_key].shape == model_state[tgt_key].shape:
                    compatible[tgt_key] = pretrained[src_key]
                else:
                    skipped.append(f"{src_key}->{tgt_key}: {pretrained[src_key].shape} vs {model_state[tgt_key].shape}")

        result = self.model.load_state_dict(compatible, strict=False)
        log.info(
            "Loaded %d/%d pretrained weights (%d skipped, %d missing)",
            len(compatible), len(pretrained), len(skipped), len(result.missing_keys),
        )
        if skipped:
            log.info("Skipped (shape mismatch): %s", skipped[:10])
        if result.missing_keys:
            log.info("Missing keys (expected for new heads): %s", result.missing_keys[:15])

    def _build_key_mapping(self, pretrained):
        """Build mapping from vendor checkpoint keys to our model keys.

        Vendor keys look like:
            encoder.patch_embedding.shallownet.0.weight
            embedder.embed_model.model.0.weight
            embedder.msk_embed
            decoder.transformer.h.0.attn.c_attn.weight
            unembedder.model.0.weight

        Our model has the same structure nested under self.model:
            model.encoder.patch_embedding.shallownet.0.weight
            model.embedder.embed_model.model.0.weight
            ...
        """
        key_map = {}
        model_state = self.model.state_dict()

        for src_key in pretrained:
            # Direct mapping: vendor key -> model.<vendor key>
            tgt_key = src_key
            if tgt_key in model_state:
                key_map[src_key] = tgt_key
            else:
                # Try without any prefix transformations
                # The vendor checkpoint keys should map directly since we used the same
                # attribute names (encoder, embedder, decoder, unembedder)
                pass

        log.info("Key mapping: %d/%d vendor keys mapped", len(key_map), len(pretrained))
        return key_map

    def forward(self, chunks, attention_mask):
        """
        Args:
            chunks: (batch, num_chunks, C_input, chunk_len)
            attention_mask: (batch, num_chunks)
        """
        if self.bridge is not None:
            # Apply bridge to each chunk: (batch, num_chunks, C_in, T) -> (batch, num_chunks, 22, T)
            b, nc, c, t = chunks.size()
            chunks_flat = chunks.reshape(b * nc, c, t)
            chunks_flat = self.bridge(chunks_flat)
            chunks = chunks_flat.reshape(b, nc, NEUROGPT_N_CHANS, t)

        return self.model(chunks, attention_mask)

    def _shared_step(self, batch):
        chunks, attn_mask, y = batch
        logits = self(chunks, attn_mask)
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------

def seed_everything_deterministic(seed: int):
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
    mode: str,
    training_mode: str,
    data_dir: Path,
    output_dir: Path,
    wandb_entity: str = "braindecode",
    wandb_project: str = "adapter-finetuning",
    fast_dev_run: bool = False,
    max_chunks: int = 4,
):
    try:
        import numpy as _np
        safe_globals = [_np.dtype, _np.ndarray]
        try:
            safe_globals.append(_np._core.multiarray.scalar)
        except AttributeError:
            pass
        torch.serialization.add_safe_globals(safe_globals)
    except Exception:
        pass

    train_config = TRAIN_CONFIG_SFT if training_mode == "sft" else TRAIN_CONFIG_PROBE
    config = DATASET_CONFIG[dataset_name]

    # Determine HDF5 path
    if mode == "conv1d":
        h5_path = data_dir / f"{dataset_name}_native_128hz.h5"
    elif mode == "interpolated":
        h5_path = data_dir / f"{dataset_name}_interpolated_spline.h5"
    elif mode == "omneeg":
        h5_path = data_dir / f"{dataset_name}_omneeg_3d.h5"
    elif mode == "riemannian":
        h5_path = data_dir / f"{dataset_name}_interpolated_spline_recenter_riemannian.h5"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not h5_path.exists():
        raise FileNotFoundError(f"Data not found: {h5_path}")

    log.info("=" * 60)
    log.info(
        "Neuro-GPT | Dataset: %s, Seed: %d, Mode: %s, Training: %s",
        dataset_name, seed, mode, training_mode,
    )
    log.info("=" * 60)

    seed_everything_deterministic(seed)

    # Load dataset with resampling to 250 Hz and chunking
    dataset = NeuroGPTDataset(
        h5_path, mode=mode,
        chunk_len=NEUROGPT_CHUNK_LEN,
        chunk_ovlp=NEUROGPT_CHUNK_OVLP,
        max_chunks=max_chunks,
        target_sfreq=NEUROGPT_SFREQ,
    )

    train_ds, val_ds, test_ds = split_dataset(dataset, config, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # All modes use bridge since we never have exactly 22 Neuro-GPT channels
    use_bridge = True

    model = NeuroGPTExperimentModule(
        n_chans_input=dataset.n_channels,
        n_outputs=config["n_classes"],
        num_chunks=max_chunks,
        use_bridge=use_bridge,
        training_mode=training_mode,
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        warmup_epochs=train_config["warmup_epochs"],
        max_epochs=train_config["max_epochs"],
        eta_min=train_config["eta_min"],
    )

    experiment_name = f"neurogpt_{mode}_{training_mode}_{dataset_name}_init{seed}"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    run_name = f"{experiment_name}_job{slurm_job_id}" if slurm_job_id else experiment_name

    tags = ["neurogpt", mode, training_mode, dataset_name, f"init{seed}", "alignment_experiment"]
    if slurm_job_id:
        tags.append(f"slurm_{slurm_job_id}")

    wandb_logger = WandbLogger(
        entity=wandb_entity, project=wandb_project, name=run_name,
        group=f"neurogpt_{mode}_{training_mode}_{dataset_name}",
        tags=tags, save_dir=str(output_dir),
        config={
            "model": {"name": "neurogpt", "variant": "base"},
            "data": {"name": dataset_name, "n_classes": config["n_classes"], "batch_size": config["batch_size"]},
            "adapter": {"name": training_mode},
            "experiment": {"init_seed": seed},
            "preprocessing": f"neurogpt_{mode}",
            "training_mode": training_mode,
            "use_bridge": use_bridge,
            "n_chans_input": dataset.n_channels,
            "n_chans_model": NEUROGPT_N_CHANS,
            "chunk_len": NEUROGPT_CHUNK_LEN,
            "max_chunks": max_chunks,
            "sfreq": NEUROGPT_SFREQ,
            **train_config,
        },
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints" / experiment_name),
            monitor="val_accuracy", mode="max", save_top_k=3,
            save_last=True, save_weights_only=True,
            filename="{epoch}-{val_accuracy:.4f}",
        ),
        EarlyStopping(
            monitor="val_loss", patience=train_config["patience"],
            mode="min", min_delta=0.001,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        accelerator="auto", devices=1, precision=32,
        gradient_clip_val=train_config["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        callbacks=callbacks, logger=wandb_logger,
        deterministic="warn", fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, train_loader, val_loader)

    best_score = 0.0
    if isinstance(trainer.checkpoint_callback, ModelCheckpoint) and trainer.checkpoint_callback.best_model_score is not None:
        score = trainer.checkpoint_callback.best_model_score
        best_score = score.item() if isinstance(score, torch.Tensor) else float(score)

    if test_loader is not None and len(test_ds) > 0:
        try:
            trainer.test(model, test_loader, ckpt_path="best")
        except Exception as e:
            log.warning("Test step failed: %s", e)

    log.info("Best val accuracy: %.4f", best_score)
    return best_score


def main():
    parser = argparse.ArgumentParser(description="Train Neuro-GPT on EEG datasets")
    parser.add_argument("--dataset", type=str, nargs="+", default=["bcic2a"],
                        choices=["bcic2a", "physionet", "tuev", "faced", "mdd_mumtaz2016"])
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed index (default: 0). Runs seeds start_seed..start_seed+n_seeds-1")
    parser.add_argument("--mode", type=str, default="interpolated",
                        choices=["conv1d", "interpolated", "omneeg", "riemannian"])
    parser.add_argument("--training-mode", type=str, default="probe",
                        choices=["probe", "sft"])
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE_DIR)
    parser.add_argument("--interp-dir", type=Path, default=DEFAULT_INTERP_DIR)
    parser.add_argument("--omneeg-dir", type=Path, default=DEFAULT_OMNEEG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-chunks", type=int, default=4,
                        help="Maximum number of chunks to split each trial into (default: 4)")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="braindecode")
    parser.add_argument("--wandb-project", type=str, default="adapter-finetuning")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", "/expanse/projects/nemar/eeg_finetuning/.cache/tmp")
    os.environ.setdefault("HF_HOME", "/expanse/projects/nemar/dtyoung/huggingface_cache")
    os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))

    if args.mode == "conv1d":
        data_dir = args.native_dir
    elif args.mode in ("interpolated", "riemannian"):
        data_dir = args.interp_dir
    elif args.mode == "omneeg":
        data_dir = args.omneeg_dir
    else:
        data_dir = args.native_dir

    results = {}
    for dataset_name in args.dataset:
        dataset_results = []
        for seed in range(args.start_seed, args.start_seed + args.n_seeds):
            try:
                score = run_experiment(
                    dataset_name=dataset_name, seed=seed, mode=args.mode,
                    training_mode=args.training_mode,
                    data_dir=data_dir, output_dir=args.output_dir,
                    wandb_entity=args.wandb_entity, wandb_project=args.wandb_project,
                    fast_dev_run=args.fast_dev_run,
                    max_chunks=args.max_chunks,
                )
                dataset_results.append(score)
            except Exception as e:
                log.error("Failed %s seed %d: %s", dataset_name, seed, e, exc_info=True)
                dataset_results.append(0.0)

        results[dataset_name] = dataset_results
        log.info("%s: mean=%.4f, std=%.4f", dataset_name, np.mean(dataset_results), np.std(dataset_results))

    log.info("=" * 60)
    log.info("Neuro-GPT Results Summary (mode=%s, training=%s)", args.mode, args.training_mode)
    log.info("=" * 60)
    for dataset_name, scores in results.items():
        log.info("  %s: %.4f +/- %.4f (n=%d)", dataset_name, np.mean(scores), np.std(scores), len(scores))


if __name__ == "__main__":
    main()
