#!/usr/bin/env python
"""
Pre-compute OmnEEG 3D spherical harmonics features for EEG datasets.

Transforms raw EEG into spatially-aligned spherical harmonic coefficients
using OmnEEG's Interpolate transform. With l_max=4 (resolution=4), this
produces 25 coefficients per time point, providing a standardized spatial
representation regardless of the original channel montage.

Output format: HDF5 files with:
  - signals: [n_samples, 25, n_times]
  - labels: [n_samples]
  - metadata: dataset info, transform params, montage details

Usage:
    # Process all datasets
    python scripts/preprocess_omneeg.py

    # Process specific dataset
    python scripts/preprocess_omneeg.py --dataset bcic2a

    # Custom output directory and resolution
    python scripts/preprocess_omneeg.py --output-dir /path/to/output --resolution 4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent  # repo root

# OmnEEG is not a pip-installable package; add its clone location to sys.path
_OMNEEG_DIR = Path(__file__).resolve().parent.parent / "vendor" / "OmnEEG"
if _OMNEEG_DIR.is_dir():
    sys.path.insert(0, str(_OMNEEG_DIR))
else:
    # Fallback: check environment variable or common Expanse location
    _OMNEEG_DIR = Path(
        os.environ.get("OMNEEG_DIR", str(_REPO / "vendor/OmnEEG"))
    )
    if _OMNEEG_DIR.is_dir():
        sys.path.insert(0, str(_OMNEEG_DIR))

import h5py
import mne
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Channel name normalization (old 10/20 names -> modern names)
CHANNEL_RENAME_MAP = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "A1": "M1",
    "A2": "M2",
}


# Default paths (Expanse cluster)
DEFAULT_OUTPUT_DIR = (_REPO / "data/omneeg")
DEFAULT_TUEV_PATH = str(_REPO / "data" / "raw" / "tuev")
DEFAULT_CACHE_DIR = str(_REPO / ".cache")


# ---------------------------------------------------------------------------
# Dataset loaders — mirror the logic in adapter_finetuning data configs
# ---------------------------------------------------------------------------

def load_bcic2a():
    """Load BCI Competition IV 2a via MOABB.

    Returns list of (mne.Epochs, labels, subject_id) per subject.
    """
    from moabb.datasets import BNCI2014_001

    dataset = BNCI2014_001()
    subjects = dataset.subject_list  # [1..9]

    all_epochs = []
    all_labels = []
    all_subjects = []

    event_id = {"feet": 0, "left_hand": 1, "right_hand": 2, "tongue": 3}

    for subj in subjects:
        log.info("Loading BCI2a subject %d", subj)
        data = dataset.get_data(subjects=[subj])

        for session_name, session_data in data[subj].items():
            for run_name, raw in session_data.items():
                # Pick EEG channels only
                raw = raw.copy().pick("eeg")

                # Resample to 200 Hz (matching data config)
                if raw.info["sfreq"] != 200:
                    raw.resample(200)

                # High-pass filter
                raw.filter(l_freq=0.1, h_freq=None)

                # Create epochs from events
                events, existing_event_id = mne.events_from_annotations(raw)
                # Map MOABB event names to our class indices
                mapped_event_id = {}
                for name, code in existing_event_id.items():
                    name_lower = name.lower().replace(" ", "_")
                    if name_lower in event_id:
                        mapped_event_id[name] = code

                if not mapped_event_id:
                    continue

                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=mapped_event_id,
                    tmin=0,
                    tmax=4.0 - 1.0 / raw.info["sfreq"],
                    baseline=None,
                    preload=True,
                )

                # Map event codes to class indices
                code_to_label = {
                    code: event_id[name.lower().replace(" ", "_")]
                    for name, code in mapped_event_id.items()
                }
                labels = [code_to_label[ev] for ev in epochs.events[:, 2]]

                all_epochs.append(epochs)
                all_labels.extend(labels)
                all_subjects.extend([subj] * len(labels))

    return all_epochs, np.array(all_labels), np.array(all_subjects)


def load_physionet():
    """Load PhysioNet Motor Imagery via MOABB."""
    from moabb.datasets import PhysionetMI

    dataset = PhysionetMI()
    subjects = dataset.subject_list  # 1..109

    all_epochs = []
    all_labels = []
    all_subjects = []

    event_id = {"left_hand": 0, "right_hand": 1, "feet": 2, "hands": 3}

    # Channel subset matching physionet.yaml config
    ch_names = [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        "Fp1", "Fpz", "Fp2",
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POz", "PO4", "PO8",
        "O1", "Oz", "O2", "Iz",
    ]

    for subj in subjects:
        log.info("Loading PhysioNet subject %d/%d", subj, len(subjects))
        try:
            data = dataset.get_data(subjects=[subj])
        except Exception as e:
            log.warning("Failed to load subject %d: %s", subj, e)
            continue

        for session_name, session_data in data[subj].items():
            for run_name, raw in session_data.items():
                raw = raw.copy().pick("eeg")

                # Pick channels that exist
                available = [ch for ch in ch_names if ch in raw.ch_names]
                if len(available) < 20:
                    log.warning(
                        "Subject %d: only %d/%d channels available, skipping",
                        subj, len(available), len(ch_names),
                    )
                    continue
                raw.pick(available)

                if raw.info["sfreq"] != 200:
                    raw.resample(200)

                raw.filter(l_freq=0.1, h_freq=None)

                events, existing_event_id = mne.events_from_annotations(raw)
                mapped_event_id = {}
                for name, code in existing_event_id.items():
                    name_lower = name.lower().replace(" ", "_")
                    if name_lower in event_id:
                        mapped_event_id[name] = code

                if not mapped_event_id:
                    continue

                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=mapped_event_id,
                    tmin=0,
                    tmax=3.0 - 1.0 / raw.info["sfreq"],
                    baseline=None,
                    preload=True,
                )

                code_to_label = {
                    code: event_id[name.lower().replace(" ", "_")]
                    for name, code in mapped_event_id.items()
                }
                labels = [code_to_label[ev] for ev in epochs.events[:, 2]]

                all_epochs.append(epochs)
                all_labels.extend(labels)
                all_subjects.extend([subj] * len(labels))

    return all_epochs, np.array(all_labels), np.array(all_subjects)


def load_tuev(tuev_path: str):
    """Load TUH EEG Events (TUEV) dataset.

    Mirrors tuev.yaml: 21 standard 10-20 channels, 200 Hz, 5s windows.
    TUEV v2.0.1 uses .rec files with numeric labels:
        1:spsw, 2:gped, 3:pled, 4:eyem, 5:artf, 6:bckg
    Format: channel_id,onset,offset,label_id
    """
    ch_names = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "A1", "A2", "Fz", "Cz", "Pz",
    ]

    # .rec numeric labels -> class index (0-based for training)
    rec_label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    # spsw=0, gped=1, pled=2, eyem=3, artf=4, bckg=5

    all_epochs = []
    all_labels = []
    all_subjects = []

    log.info("Loading TUEV from %s", tuev_path)

    for split in ["train", "eval"]:
        split_dir = Path(tuev_path) / "v2.0.1" / "edf" / split
        if not split_dir.exists():
            log.warning("Split directory not found: %s", split_dir)
            continue

        edf_files = sorted(split_dir.rglob("*.edf"))
        log.info("Found %d EDF files in %s split", len(edf_files), split)

        for edf_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

                # Rename TUH channels: "EEG FP1-REF" -> "Fp1", etc.
                rename_map = {}
                for ch in raw.ch_names:
                    if ch.startswith("EEG ") and ch.endswith("-REF"):
                        short = ch.replace("EEG ", "").replace("-REF", "").strip()
                        # Capitalize properly: FP1->Fp1, FZ->Fz, CZ->Cz, PZ->Pz
                        if len(short) > 1 and short[1:].isdigit():
                            short = short[0].upper() + short[1:]
                        elif len(short) == 2 and short[1].isalpha():
                            short = short[0].upper() + short[1].lower()
                        else:
                            short = short.capitalize()
                        rename_map[ch] = short
                if rename_map:
                    raw.rename_channels(rename_map)

                # Pick available channels from our target list
                available = [ch for ch in ch_names if ch in raw.ch_names]
                if len(available) < 15:
                    continue
                raw.pick(available)

                if raw.info["sfreq"] != 200:
                    raw.resample(200)

                raw.filter(l_freq=0.1, h_freq=None)

                # Parse .rec annotation file (TUEV v2.0.1 format)
                rec_path = edf_path.with_suffix(".rec")
                if not rec_path.exists():
                    continue

                # Parse .rec: channel_id,onset,offset,label_id
                # Merge across channels: take the majority label per time window
                from collections import defaultdict
                window_labels = defaultdict(list)
                with open(rec_path) as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) != 4:
                            continue
                        try:
                            onset = float(parts[1])
                            offset = float(parts[2])
                            label_id = int(parts[3])
                            if label_id in rec_label_map:
                                # Group by (onset, offset) to merge across channels
                                window_labels[(onset, offset)].append(label_id)
                        except ValueError:
                            continue

                if not window_labels:
                    continue

                # Deduplicate: take majority vote per time window
                annotations = []
                for (onset, offset), labels in window_labels.items():
                    from collections import Counter
                    majority_label = Counter(labels).most_common(1)[0][0]
                    annotations.append((onset, offset, majority_label))

                # Create fixed-length windows (5s) from annotations
                sfreq = raw.info["sfreq"]
                for onset, offset, label_id in annotations:
                    duration = offset - onset
                    # Each .rec annotation is typically 1s; stride 5s windows
                    # Use onset directly as window start (no 2s offset for .rec)
                    if onset + 5.0 > raw.times[-1]:
                        continue

                    n_samples = int(5.0 * sfreq)  # Fixed window size
                    start_samp = int(onset * sfreq)
                    stop_samp = start_samp + n_samples

                    if stop_samp > raw.n_times:
                        continue

                    data = raw.get_data(start=start_samp, stop=stop_samp)
                    info = raw.info.copy()
                    epoch = mne.EpochsArray(
                        data[np.newaxis, :, :], info, verbose=False
                    )

                    all_epochs.append(epoch)
                    all_labels.append(rec_label_map[label_id])

                    # Subject ID from directory name; prefix with split
                    subj_id = f"{split}/{edf_path.parent.name}"
                    all_subjects.append(subj_id)

            except Exception as e:
                log.warning("Failed to load %s: %s", edf_path.name, e)
                continue

    return all_epochs, np.array(all_labels), np.array(all_subjects)


def load_faced(data_root=str(_REPO / "data" / "raw" / "faced"),
               target_sfreq=200.0):
    """Load FACED emotion dataset from BIDS format.

    26 EEG channels, 9-class emotion, 10s windows at 200Hz.
    Subjects: sub-000 to sub-122 (123 total).
    Labels from emotion_label column in events.tsv.
    """
    import pandas as pd

    data_root = Path(data_root)
    ch_names_target = [
        "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
        "Cz", "C3", "C4", "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4",
        "PO3", "PO4", "Oz", "O1", "O2",
    ]
    emotion_classes = {
        "Anger": 0, "Disgust": 1, "Fear": 2, "Sadness": 3, "Neutral": 4,
        "Amusement": 5, "Inspiration": 6, "Joy": 7, "Tenderness": 8,
    }
    window_sec = 10.0

    all_epochs = []
    all_labels = []
    all_subjects = []

    subject_dirs = sorted(data_root.glob("sub-*"))
    log.info("FACED: found %d subject directories", len(subject_dirs))

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name  # e.g. "sub-001"
        subj_num = int(subj_id.split("-")[1])
        eeg_dir = subj_dir / "eeg"

        bdf_files = sorted(eeg_dir.glob("*.bdf"))
        if not bdf_files:
            log.warning("No BDF files found for %s", subj_id)
            continue

        for bdf_path in bdf_files:
            try:
                raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
                raw = raw.copy().pick("eeg")

                # Rename old-style channel names
                rename = {}
                for ch in raw.ch_names:
                    for old, new in CHANNEL_RENAME_MAP.items():
                        if ch == old:
                            rename[ch] = new
                if rename:
                    raw.rename_channels(rename)

                # Pick target channels (use modern names after rename)
                modern_target = []
                for ch in ch_names_target:
                    modern_ch = CHANNEL_RENAME_MAP.get(ch, ch)
                    if modern_ch in raw.ch_names:
                        modern_target.append(modern_ch)
                    elif ch in raw.ch_names:
                        modern_target.append(ch)

                if len(modern_target) < 20:
                    log.warning("%s: only %d/%d channels available, skipping",
                                subj_id, len(modern_target), len(ch_names_target))
                    continue
                raw.pick(modern_target)

                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq)
                raw.filter(l_freq=0.1, h_freq=None)

                # Read events from events.tsv
                events_tsv = bdf_path.with_name(
                    bdf_path.name.replace("_eeg.bdf", "_events.tsv")
                )
                if not events_tsv.exists():
                    log.warning("No events.tsv for %s", bdf_path.name)
                    continue

                events_df = pd.read_csv(events_tsv, sep="\t")

                # Filter to emotion-labeled events
                if "emotion_label" not in events_df.columns:
                    log.warning("No emotion_label column in %s", events_tsv)
                    continue

                for _, row in events_df.iterrows():
                    emotion = row.get("emotion_label", "n/a")
                    if emotion not in emotion_classes:
                        continue

                    onset = float(row["onset"])
                    duration = float(row["duration"]) if "duration" in row and str(row["duration"]) != "n/a" else 0.0
                    if duration <= 0:
                        continue

                    sfreq = raw.info["sfreq"]
                    n_windows = int(duration / window_sec)
                    label = emotion_classes[emotion]

                    for w in range(n_windows):
                        start = onset + w * window_sec
                        start_samp = int(start * sfreq)
                        stop_samp = int((start + window_sec) * sfreq)

                        if stop_samp > raw.n_times:
                            break

                        data = raw.get_data(start=start_samp, stop=stop_samp)
                        info = raw.info.copy()
                        epoch = mne.EpochsArray(
                            data[np.newaxis, :, :], info, verbose=False
                        )
                        all_epochs.append(epoch)
                        all_labels.append(label)
                        all_subjects.append(subj_num)

            except Exception as e:
                log.warning("Failed to load %s: %s", bdf_path.name, e)
                continue

    log.info("FACED: loaded %d epochs from %d subjects",
             len(all_epochs), len(set(all_subjects)))
    return all_epochs, np.array(all_labels), np.array(all_subjects)


def load_isruc_sleep(data_root=str(_REPO / "data" / "raw" / "isruc-sleep"),
                     target_sfreq=200.0):
    """Load ISRUC-Sleep dataset from BIDS format.

    6 bipolar EEG channels, 5-class sleep staging, 30s windows at 200Hz.
    Bipolar channels: C3-A2 -> C3-M2, C4-A1 -> C4-M1, F3-A2 -> F3-M2,
                      F4-A1 -> F4-M1, O1-A2 -> O1-M2, O2-A1 -> O2-M1
    Proxy 10-20 positions: O1, O2, AF8, P8, PO10, PO9
    """
    import pandas as pd

    data_root = Path(data_root)
    # The bipolar channels as they appear in the raw EDF files
    bipolar_channels = ["C3-A2", "C4-A1", "F3-A2", "F4-A1", "O1-A2", "O2-A1"]
    # Modern bipolar names (after A->M rename)
    bipolar_modern = ["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]
    # Proxy 10-20 channel positions for interpolation/montage
    proxy_positions = ["O1", "O2", "AF8", "P8", "PO10", "PO9"]

    sleep_classes = {
        "Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2,
        "Sleep stage N3": 3, "Sleep stage R": 4,
    }
    window_sec = 30.0

    all_epochs = []
    all_labels = []
    all_subjects = []

    subject_dirs = sorted(data_root.glob("sub-*"))
    log.info("ISRUC-Sleep: found %d subject directories", len(subject_dirs))

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name  # e.g. "sub-I001"
        subj_short = subj_id.split("-")[1]  # e.g. "I001"
        eeg_dir = subj_dir / "eeg"
        if not eeg_dir.exists():
            continue

        edf_files = sorted(eeg_dir.glob("*_eeg.edf"))
        if not edf_files:
            continue

        for edf_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

                # Check if bipolar channels exist (try both A and M naming)
                available_bipolar = []
                raw_ch_set = set(raw.ch_names)
                for bp_old, bp_new in zip(bipolar_channels, bipolar_modern):
                    if bp_new in raw_ch_set:
                        available_bipolar.append(bp_new)
                    elif bp_old in raw_ch_set:
                        available_bipolar.append(bp_old)

                if len(available_bipolar) < 6:
                    log.warning("%s: only %d/6 bipolar channels, skipping",
                                subj_id, len(available_bipolar))
                    continue

                raw.pick(available_bipolar)

                # Rename A->M for consistency
                rename = {}
                for ch in raw.ch_names:
                    new_ch = ch.replace("-A2", "-M2").replace("-A1", "-M1")
                    if new_ch != ch:
                        rename[ch] = new_ch
                if rename:
                    raw.rename_channels(rename)

                # Now rename bipolar channels to proxy 10-20 positions
                bp_to_proxy = dict(zip(bipolar_modern, proxy_positions))
                rename_proxy = {}
                for ch in raw.ch_names:
                    if ch in bp_to_proxy:
                        rename_proxy[ch] = bp_to_proxy[ch]
                if rename_proxy:
                    raw.rename_channels(rename_proxy)

                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq)
                raw.filter(l_freq=0.1, h_freq=None)

                # Read events from events.tsv
                events_tsv = edf_path.with_name(
                    edf_path.name.replace("_eeg.edf", "_events.tsv")
                )
                if not events_tsv.exists():
                    log.warning("No events.tsv for %s", edf_path.name)
                    continue

                events_df = pd.read_csv(events_tsv, sep="\t")
                sfreq = raw.info["sfreq"]

                for _, row in events_df.iterrows():
                    trial_type = str(row.get("trial_type", ""))
                    if trial_type not in sleep_classes:
                        continue

                    onset = float(row["onset"])
                    duration = float(row.get("duration", window_sec))
                    label = sleep_classes[trial_type]

                    start_samp = int(onset * sfreq)
                    stop_samp = int((onset + window_sec) * sfreq)

                    if stop_samp > raw.n_times:
                        break

                    data = raw.get_data(start=start_samp, stop=stop_samp)
                    info = raw.info.copy()
                    epoch = mne.EpochsArray(
                        data[np.newaxis, :, :], info, verbose=False
                    )
                    all_epochs.append(epoch)
                    all_labels.append(label)
                    all_subjects.append(subj_short)

            except Exception as e:
                log.warning("Failed to load %s: %s", edf_path.name, e)
                continue

    log.info("ISRUC-Sleep: loaded %d epochs from %d subjects",
             len(all_epochs), len(set(all_subjects)))
    return all_epochs, np.array(all_labels), np.array(all_subjects)


def load_mdd_mumtaz(data_root=str(_REPO / "data" / "raw" / "mdd_mumtaz2016"),
                    target_sfreq=200.0):
    """Load MDD Mumtaz 2016 dataset from BIDS format.

    19 EEG channels, 2-class (healthy vs MDD), 5s windows at 200Hz.
    Labels from subject ID: HS* = healthy (0), MDDS* = MDD (1).
    Uses eyesClosed task recordings.
    """
    data_root = Path(data_root)
    ch_names_target = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T3", "C3", "Cz", "C4", "T4",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2",
    ]
    window_sec = 5.0

    all_epochs = []
    all_labels = []
    all_subjects = []

    subject_dirs = sorted(data_root.glob("sub-*"))
    log.info("MDD Mumtaz: found %d subject directories", len(subject_dirs))

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name  # e.g. "sub-HS1" or "sub-MDDS10"
        subj_short = subj_id.split("-", 1)[1]  # e.g. "HS1" or "MDDS10"

        # Determine label from subject ID
        if subj_short.startswith("HS"):
            label = 0  # healthy
        elif subj_short.startswith("MDDS"):
            label = 1  # MDD
        else:
            log.warning("Unknown subject pattern: %s", subj_id)
            continue

        eeg_dir = subj_dir / "eeg"
        if not eeg_dir.exists():
            continue

        # Prefer eyesClosed task, also try eyesOpen
        edf_files = sorted(eeg_dir.glob("*_task-eyesClosed_eeg.edf"))
        if not edf_files:
            edf_files = sorted(eeg_dir.glob("*_task-eyesOpen_eeg.edf"))
        if not edf_files:
            edf_files = sorted(eeg_dir.glob("*_eeg.edf"))
        if not edf_files:
            log.warning("No EDF files found for %s", subj_id)
            continue

        for edf_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                raw = raw.copy().pick("eeg")

                # Strip EEG prefix and reference suffix (e.g., "EEG Fp1-LE" -> "Fp1")
                rename_ref = {}
                for ch in raw.ch_names:
                    clean = ch
                    # Strip "EEG " prefix
                    if clean.startswith("EEG "):
                        clean = clean[4:]
                    # Strip reference suffix (e.g., "-LE", "-RE", "-Ref")
                    if "-" in clean:
                        clean = clean.split("-")[0]
                    clean = clean.strip()
                    if clean != ch:
                        rename_ref[ch] = clean
                if rename_ref:
                    raw.rename_channels(rename_ref)

                # Rename old-style channel names
                rename = {}
                for ch in raw.ch_names:
                    for old, new in CHANNEL_RENAME_MAP.items():
                        if ch == old:
                            rename[ch] = new
                if rename:
                    raw.rename_channels(rename)

                # Pick target channels (try with modern names)
                available = []
                for ch in ch_names_target:
                    modern_ch = CHANNEL_RENAME_MAP.get(ch, ch)
                    if modern_ch in raw.ch_names:
                        available.append(modern_ch)
                    elif ch in raw.ch_names:
                        available.append(ch)

                if len(available) < 15:
                    log.warning("%s: only %d/%d channels available, skipping",
                                subj_id, len(available), len(ch_names_target))
                    continue
                raw.pick(available)

                if raw.info["sfreq"] != target_sfreq:
                    raw.resample(target_sfreq)
                raw.filter(l_freq=0.1, h_freq=None)

                # Create fixed-length windows
                sfreq = raw.info["sfreq"]
                n_samples_per_window = int(window_sec * sfreq)
                total_samples = raw.n_times
                n_windows = total_samples // n_samples_per_window

                for w in range(n_windows):
                    start_samp = w * n_samples_per_window
                    stop_samp = start_samp + n_samples_per_window

                    data = raw.get_data(start=start_samp, stop=stop_samp)
                    info = raw.info.copy()
                    epoch = mne.EpochsArray(
                        data[np.newaxis, :, :], info, verbose=False
                    )
                    all_epochs.append(epoch)
                    all_labels.append(label)
                    all_subjects.append(subj_short)

            except Exception as e:
                log.warning("Failed to load %s: %s", edf_path.name, e)
                continue

    log.info("MDD Mumtaz: loaded %d epochs from %d subjects",
             len(all_epochs), len(set(all_subjects)))
    return all_epochs, np.array(all_labels), np.array(all_subjects)



# ---------------------------------------------------------------------------
# OmnEEG transform
# ---------------------------------------------------------------------------

def apply_omneeg_transform(epochs_list, resolution=4):
    """Apply OmnEEG 3D spherical harmonics transform to a list of Epochs.

    Parameters
    ----------
    epochs_list : list of mne.Epochs
        Each element may contain one or more epochs.
    resolution : int
        Spherical harmonics resolution (l_max). With resolution=4,
        n_coeffs = (4+1)^2 = 25.

    Returns
    -------
    signals : np.ndarray
        Shape [n_total_epochs, n_coeffs, n_times].
    """
    from omneeg.transform import Interpolate

    transform = Interpolate(resolution=resolution, transform_type="3d")
    n_coeffs = (resolution + 1) ** 2  # 25 for resolution=4

    all_signals = []

    for i, epochs in enumerate(epochs_list):
        if i % 100 == 0:
            log.info("Transforming epoch batch %d/%d", i, len(epochs_list))

        try:
            # OmnEEG expects mne.Epochs; apply transform
            coeffs = transform(epochs)
            # Expected shape: [n_epochs, n_coeffs, n_times]
            if coeffs.ndim == 2:
                coeffs = coeffs[np.newaxis, :, :]

            if coeffs.shape[1] != n_coeffs:
                log.warning(
                    "Unexpected n_coeffs: %d (expected %d), skipping",
                    coeffs.shape[1], n_coeffs,
                )
                continue

            all_signals.append(coeffs)
        except Exception as e:
            log.warning("Transform failed for epoch batch %d: %s", i, e)
            continue

    if not all_signals:
        raise RuntimeError("No epochs were successfully transformed")

    # Truncate all epochs to minimum time dimension
    min_t = min(s.shape[2] for s in all_signals)
    all_signals = [s[:, :, :min_t] for s in all_signals]
    
    return np.concatenate(all_signals, axis=0)


def ensure_montage(epochs):
    """Ensure epochs have a standard montage set for OmnEEG."""
    if epochs.get_montage() is None:
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            epochs.set_montage(montage, on_missing="warn")
        except Exception as e:
            log.warning("Could not set montage: %s", e)
    return epochs


# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------

def save_to_hdf5(
    filepath: Path,
    signals: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    dataset_name: str,
    resolution: int,
    sfreq: float,
):
    """Save OmnEEG features to HDF5.

    File structure:
        signals: [n_samples, n_coeffs, n_times] float32
        labels: [n_samples] int64
        subjects: [n_samples] (int or string)
        attrs: dataset_name, resolution, n_coeffs, sfreq, n_samples
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("signals", data=signals.astype(np.float32), compression="gzip")
        f.create_dataset("labels", data=labels.astype(np.int64))

        # Subjects may be strings (TUEV) or ints
        if subjects.dtype.kind in ("U", "S", "O"):
            f.create_dataset("subjects", data=[s.encode('utf-8') for s in subjects.astype(str)])
        else:
            f.create_dataset("subjects", data=subjects)

        # Metadata
        f.attrs["dataset_name"] = dataset_name
        f.attrs["resolution"] = resolution
        f.attrs["n_coeffs"] = (resolution + 1) ** 2
        f.attrs["sfreq"] = sfreq
        f.attrs["n_samples"] = signals.shape[0]
        f.attrs["n_times"] = signals.shape[2]

    log.info(
        "Saved %s: %d samples, shape %s -> %s",
        filepath, signals.shape[0], signals.shape, filepath,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "bcic2a": load_bcic2a,
    "physionet": load_physionet,
    "tuev": None,  # Needs path argument
    "faced": load_faced,
    "isruc-sleep": load_isruc_sleep,
    "mdd_mumtaz2016": load_mdd_mumtaz,
}

DATASET_SFREQ = {
    "bcic2a": 200.0,
    "physionet": 200.0,
    "tuev": 200.0,
    "faced": 200.0,
    "isruc-sleep": 200.0,
    "mdd_mumtaz2016": 200.0,
}


def process_dataset(dataset_name: str, output_dir: Path, resolution: int, tuev_path: str):
    """Process a single dataset end-to-end."""
    log.info("=" * 60)
    log.info("Processing dataset: %s", dataset_name)
    log.info("=" * 60)

    # Load raw epochs
    if dataset_name == "tuev":
        epochs_list, labels, subjects = load_tuev(tuev_path)
    else:
        loader = DATASET_LOADERS[dataset_name]
        epochs_list, labels, subjects = loader()

    log.info("Loaded %d epoch batches, %d total labels", len(epochs_list), len(labels))

    if len(epochs_list) == 0:
        log.error("No data loaded for %s", dataset_name)
        return

    # Ensure montages are set
    epochs_list = [ensure_montage(ep) for ep in epochs_list]

    # Apply OmnEEG transform
    log.info("Applying OmnEEG 3D spherical harmonics (resolution=%d)...", resolution)
    signals = apply_omneeg_transform(epochs_list, resolution=resolution)

    # Verify shape
    n_coeffs = (resolution + 1) ** 2
    assert signals.shape[0] == len(labels), (
        f"Mismatch: {signals.shape[0]} signals vs {len(labels)} labels"
    )
    assert signals.shape[1] == n_coeffs, (
        f"Unexpected n_coeffs: {signals.shape[1]} (expected {n_coeffs})"
    )

    log.info("Transformed signals shape: %s", signals.shape)

    # Save
    output_path = output_dir / f"{dataset_name}_omneeg_3d.h5"
    save_to_hdf5(
        output_path,
        signals,
        labels,
        subjects,
        dataset_name,
        resolution,
        DATASET_SFREQ[dataset_name],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute OmnEEG 3D spherical harmonics features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["bcic2a", "physionet", "tuev", "faced", "isruc-sleep", "mdd_mumtaz2016"],
        choices=["bcic2a", "physionet", "tuev", "faced", "isruc-sleep", "mdd_mumtaz2016"],
        help="Datasets to process (default: all three)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=4,
        help="Spherical harmonics resolution (l_max). Default: 4 -> 25 coefficients",
    )
    parser.add_argument(
        "--tuev-path",
        type=str,
        default=DEFAULT_TUEV_PATH,
        help="Path to TUEV dataset directory",
    )

    args = parser.parse_args()

    log.info("OmnEEG Preprocessing")
    log.info("Output dir: %s", args.output_dir)
    log.info("Resolution: %d (n_coeffs=%d)", args.resolution, (args.resolution + 1) ** 2)
    log.info("Datasets: %s", args.dataset)

    for dataset_name in args.dataset:
        process_dataset(dataset_name, args.output_dir, args.resolution, args.tuev_path)

    log.info("All datasets processed!")


if __name__ == "__main__":
    main()
