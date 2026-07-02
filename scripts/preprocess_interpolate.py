#!/usr/bin/env python
"""
Pre-compute MNE spherical spline interpolated EEG for BENDR's 19-channel montage.

Uses MNE's _make_interpolation_matrix (Perrin et al. 1989 spherical splines)
to project EEG data from any source montage to BENDR's expected 19 standard
10/20 channels. This is a physics-grounded channel adaptation method that
uses actual electrode 3D positions.

Output format: HDF5 files with:
  - signals: [n_samples, 19, n_times]
  - labels: [n_samples]
  - subjects: [n_samples]
  - metadata: dataset info, method, channel names

Usage:
    # Process all datasets
    python scripts/preprocess_interpolate.py

    # Process specific dataset
    python scripts/preprocess_interpolate.py --dataset bcic2a

    # Use MNE method instead of spline
    python scripts/preprocess_interpolate.py --method MNE

    # SSI + Riemannian re-centering
    python scripts/preprocess_interpolate.py --method spline --recenter riemannian
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix
from scipy.linalg import sqrtm, inv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Default paths (Expanse cluster)
DEFAULT_OUTPUT_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/interpolated")
DEFAULT_TUEV_PATH = "/expanse/projects/nemar/eeg_finetuning/data/tuh_eeg/tuh_eeg_events"
DEFAULT_CACHE_DIR = "/expanse/projects/nemar/eeg_finetuning/.cache"

# BENDR's 19 standard 10/20 EEG channels (used in TUH pretraining)
BENDR_19_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
]

# Channel name normalization (old 10/20 names -> modern names)
CHANNEL_RENAME_MAP = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "A1": "M1",
    "A2": "M2",
}


# ---------------------------------------------------------------------------
# Dataset loaders — reused from preprocess_omneeg.py
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
                raw = raw.copy().pick("eeg")

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
                    tmax=4.0 - 1.0 / raw.info["sfreq"],
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


def load_physionet():
    """Load PhysioNet Motor Imagery via MOABB."""
    from moabb.datasets import PhysionetMI

    dataset = PhysionetMI()
    subjects = dataset.subject_list  # 1..109

    all_epochs = []
    all_labels = []
    all_subjects = []

    event_id = {"left_hand": 0, "right_hand": 1, "feet": 2, "hands": 3}

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
    """Load TUH EEG Events (TUEV) dataset."""
    ch_names = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "A1", "A2", "Fz", "Cz", "Pz",
    ]

    event_id = {"spsw": 0, "gped": 1, "pled": 2, "eyem": 3, "artf": 4, "bckg": 5}

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

                # TUEV channels are named "EEG FP1-REF" etc. — strip prefix/suffix
                rename = {}
                for ch in raw.ch_names:
                    clean = ch.replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()
                    # Fix case: FP1->Fp1, FP2->Fp2, FZ->Fz, CZ->Cz, PZ->Pz
                    if clean.upper() in ("FP1", "FP2"):
                        clean = "Fp" + clean[-1]
                    elif clean.upper() in ("FZ", "CZ", "PZ"):
                        clean = clean[0].upper() + "z"
                    if clean != ch:
                        rename[ch] = clean
                if rename:
                    raw.rename_channels(rename)

                available = [ch for ch in ch_names if ch in raw.ch_names]
                if len(available) < 15:
                    continue
                raw.pick(available)

                if raw.info["sfreq"] != 200:
                    raw.resample(200)

                raw.filter(l_freq=0.1, h_freq=None)

                # TUEV v2.0.1 uses .rec files (channel,onset,offset,label_id)
                # with 1-second per-channel annotations
                rec_path = edf_path.with_suffix(".rec")
                tse_path = edf_path.with_suffix(".tse")
                tse_bi_path = edf_path.with_suffix(".tse_bi")

                # Try .rec first (v2.0.1), then .tse_bi, then .tse
                if rec_path.exists():
                    ann_path = rec_path
                    ann_format = "rec"
                elif tse_bi_path.exists():
                    ann_path = tse_bi_path
                    ann_format = "tse"
                elif tse_path.exists():
                    ann_path = tse_path
                    ann_format = "tse"
                else:
                    continue

                # Label ID mapping for .rec format
                rec_label_map = {1: "spsw", 2: "gped", 3: "pled",
                                 4: "eyem", 5: "artf", 6: "bckg"}

                if ann_format == "rec":
                    # Parse .rec: aggregate per-channel 1s annotations
                    # into recording-level segments by majority vote per second
                    from collections import Counter
                    time_labels = {}  # onset -> list of label_ids
                    with open(ann_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            # Handle [xN] prefix
                            if line.startswith("["):
                                line = line.split("]", 1)[1].strip()
                            parts = line.split(",")
                            if len(parts) == 4:
                                try:
                                    onset = float(parts[1])
                                    label_id = int(parts[3])
                                    if label_id in rec_label_map:
                                        time_labels.setdefault(onset, []).append(label_id)
                                except (ValueError, IndexError):
                                    continue

                    if not time_labels:
                        continue

                    # Majority vote per time step
                    time_majority = {}
                    for t, ids in sorted(time_labels.items()):
                        majority = Counter(ids).most_common(1)[0][0]
                        time_majority[t] = rec_label_map[majority]

                    # Merge consecutive same-label segments
                    sorted_times = sorted(time_majority.keys())
                    segments = []
                    seg_start = sorted_times[0]
                    seg_label = time_majority[seg_start]
                    seg_end = seg_start + 1.0

                    for t in sorted_times[1:]:
                        curr_label = time_majority[t]
                        if curr_label == seg_label and abs(t - seg_end) < 1.5:
                            seg_end = t + 1.0
                        else:
                            segments.append((seg_start, seg_end, seg_label))
                            seg_start = t
                            seg_label = curr_label
                            seg_end = t + 1.0
                    segments.append((seg_start, seg_end, seg_label))

                    annotations = [(s, e, l) for s, e, l in segments
                                   if l in event_id]
                else:
                    # Parse .tse format: onset offset label
                    annotations = []
                    with open(ann_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                try:
                                    onset = float(parts[0])
                                    offset = float(parts[1])
                                    label = parts[2].lower()
                                    if label in event_id:
                                        annotations.append((onset, offset, label))
                                except ValueError:
                                    continue

                if not annotations:
                    continue

                sfreq = raw.info["sfreq"]
                window_sec = 5.0
                for onset, offset, label in annotations:
                    duration = offset - onset
                    if duration < window_sec:
                        continue

                    # Extract multiple non-overlapping windows from segment
                    n_windows = int(duration / window_sec)
                    for w in range(n_windows):
                        start = onset + w * window_sec
                        if start + window_sec > raw.times[-1]:
                            break

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
                        all_labels.append(event_id[label])

                        subj_id = f"{split}_{edf_path.parent.name}"
                        all_subjects.append(subj_id)

            except Exception as e:
                log.warning("Failed to load %s: %s", edf_path.name, e)
                continue

    return all_epochs, np.array(all_labels), np.array(all_subjects)


def load_faced(data_root="/expanse/projects/nemar/eeg_finetuning/data/faced",
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


def load_isruc_sleep(data_root="/expanse/projects/nemar/eeg_finetuning/data/isruc-sleep",
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


def load_mdd_mumtaz(data_root="/expanse/projects/nemar/eeg_finetuning/data_new/mdd_mumtaz2016",
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
# Interpolation logic
# ---------------------------------------------------------------------------

def normalize_channel_names(epochs):
    """Rename old-style channel names to modern 10/20 names."""
    rename = {}
    for ch in epochs.ch_names:
        new_ch = ch
        for old, new in CHANNEL_RENAME_MAP.items():
            new_ch = new_ch.replace(old, new)
        if new_ch != ch:
            rename[ch] = new_ch
    if rename:
        epochs.rename_channels(rename)
    return epochs


def ensure_montage(epochs):
    """Ensure epochs have a standard 10/20 montage set."""
    if epochs.get_montage() is None:
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            epochs.set_montage(montage, on_missing="warn")
        except Exception as e:
            log.warning("Could not set montage: %s", e)
    return epochs


def get_channel_positions(ch_names, montage_name="standard_1020"):
    """Get 3D positions for a list of channel names from a standard montage.

    Parameters
    ----------
    ch_names : list of str
        Channel names to get positions for.
    montage_name : str
        Name of the standard montage.

    Returns
    -------
    positions : np.ndarray, shape (n_channels, 3)
        3D positions of the channels.
    valid_channels : list of str
        Channel names for which positions were found.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    all_positions = montage.get_positions()["ch_pos"]

    positions = []
    valid_channels = []
    for ch in ch_names:
        if ch in all_positions:
            positions.append(all_positions[ch])
            valid_channels.append(ch)
        else:
            log.warning("Channel %s not found in %s montage", ch, montage_name)

    return np.array(positions), valid_channels


def compute_interpolation_matrix(source_ch_names, target_ch_names, alpha=1e-5):
    """Compute spherical spline interpolation matrix from source to target channels.

    Parameters
    ----------
    source_ch_names : list of str
        Source channel names (must be in standard_1020 montage).
    target_ch_names : list of str
        Target channel names (must be in standard_1020 montage).
    alpha : float
        Regularization parameter for spherical spline interpolation.

    Returns
    -------
    interp_matrix : np.ndarray, shape (n_target, n_source)
        Matrix that maps source signals to target channel locations.
    """
    source_pos, valid_source = get_channel_positions(source_ch_names)
    target_pos, valid_target = get_channel_positions(target_ch_names)

    if len(valid_source) != len(source_ch_names):
        missing = set(source_ch_names) - set(valid_source)
        log.warning("Missing source channels in montage: %s", missing)

    if len(valid_target) != len(target_ch_names):
        missing = set(target_ch_names) - set(valid_target)
        raise ValueError(f"Missing target channels in montage: {missing}")

    log.info(
        "Computing interpolation matrix: %d source -> %d target channels",
        len(valid_source), len(valid_target),
    )

    interp_matrix = _make_interpolation_matrix(source_pos, target_pos, alpha=alpha)
    log.info("Interpolation matrix shape: %s", interp_matrix.shape)

    return interp_matrix, valid_source


def apply_interpolation(epochs_list, target_ch_names, alpha=1e-5):
    """Apply spherical spline interpolation to map epochs to target channels.

    Parameters
    ----------
    epochs_list : list of mne.Epochs
        Each element may contain one or more epochs.
    target_ch_names : list of str
        Target channel names for BENDR.
    alpha : float
        Regularization parameter.

    Returns
    -------
    signals : np.ndarray, shape (n_total_epochs, n_target, n_times)
        Interpolated signals.
    """
    all_signals = []

    # Group epochs by channel configuration to reuse interpolation matrix
    interp_cache = {}

    for i, epochs in enumerate(epochs_list):
        if i % 100 == 0:
            log.info("Interpolating epoch batch %d/%d", i, len(epochs_list))

        try:
            # Normalize channel names and set montage
            epochs = normalize_channel_names(epochs)
            epochs = ensure_montage(epochs)

            # Get source channel names (only EEG)
            source_ch_names = epochs.ch_names
            cache_key = tuple(source_ch_names)

            # Compute or retrieve interpolation matrix
            if cache_key not in interp_cache:
                interp_matrix, valid_source = compute_interpolation_matrix(
                    list(source_ch_names), target_ch_names, alpha=alpha
                )
                # Build channel index mapping (source channels that have positions)
                source_idx = [
                    list(source_ch_names).index(ch) for ch in valid_source
                ]
                interp_cache[cache_key] = (interp_matrix, source_idx)

            interp_matrix, source_idx = interp_cache[cache_key]

            # Get data: [n_epochs, n_channels, n_times]
            data = epochs.get_data()

            # Select only channels with valid montage positions
            data_valid = data[:, source_idx, :]

            # Apply interpolation: [n_target, n_source] @ [n_epochs, n_source, n_times]
            # -> [n_epochs, n_target, n_times]
            interpolated = np.einsum("ij,bjk->bik", interp_matrix, data_valid)

            all_signals.append(interpolated)

        except Exception as e:
            log.warning("Interpolation failed for epoch batch %d: %s", i, e)
            continue

    if not all_signals:
        raise RuntimeError("No epochs were successfully interpolated")

    # Truncate to minimum time length (handles off-by-one from sample rounding)
    min_t = min(s.shape[2] for s in all_signals)
    all_signals = [s[:, :, :min_t] for s in all_signals]

    return np.concatenate(all_signals, axis=0)


# ---------------------------------------------------------------------------
# Riemannian re-centering (Mellot et al., EUSIPCO 2024)
# ---------------------------------------------------------------------------

def _spd_sqrt_inv(C, reg=1e-7):
    """Compute C^{-1/2} for a symmetric positive-definite matrix C.

    Uses eigendecomposition for numerical stability.
    """
    C = (C + C.T) / 2  # ensure symmetry
    C += reg * np.eye(C.shape[0])  # regularise
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, reg)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _geometric_mean_spd(covs, tol=1e-12, max_iter=50):
    """Compute the Riemannian (Frechet) geometric mean of SPD matrices.

    Uses the iterative fixed-point algorithm (Barachant et al. 2013).

    Parameters
    ----------
    covs : np.ndarray, shape (n_matrices, n_channels, n_channels)
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    mean : np.ndarray, shape (n_channels, n_channels)
    """
    n = covs.shape[0]
    p = covs.shape[1]

    # Initialize with arithmetic mean
    mean = np.mean(covs, axis=0)
    mean = (mean + mean.T) / 2

    for iteration in range(max_iter):
        # Compute mean^{-1/2}
        mean_isqrt = _spd_sqrt_inv(mean)

        # Project to tangent space at current mean, average, project back
        S = np.zeros((p, p))
        for i in range(n):
            # S_i = mean^{-1/2} @ C_i @ mean^{-1/2}
            tmp = mean_isqrt @ covs[i] @ mean_isqrt
            # Ensure symmetry and compute log
            tmp = (tmp + tmp.T) / 2
            eigvals, eigvecs = np.linalg.eigh(tmp)
            eigvals = np.maximum(eigvals, 1e-15)
            S += eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
        S /= n

        # Check convergence
        norm = np.linalg.norm(S, 'fro')
        if norm < tol:
            break

        # Update: mean <- mean^{1/2} @ expm(S) @ mean^{1/2}
        eigvals_S, eigvecs_S = np.linalg.eigh(S)
        expS = eigvecs_S @ np.diag(np.exp(eigvals_S)) @ eigvecs_S.T

        eigvals_m, eigvecs_m = np.linalg.eigh(mean)
        eigvals_m = np.maximum(eigvals_m, 1e-15)
        mean_sqrt = eigvecs_m @ np.diag(np.sqrt(eigvals_m)) @ eigvecs_m.T

        mean = mean_sqrt @ expS @ mean_sqrt
        mean = (mean + mean.T) / 2

    return mean


def riemannian_recenter(signals, subjects, reg=1e-6):
    """Apply Riemannian re-centering: per-subject geometric mean whitening.

    For each subject, compute the Riemannian geometric mean of their trial
    covariance matrices, then whiten all trials by M^{-1/2}.

    This aligns each subject's data distribution to the identity on the
    SPD manifold, reducing inter-subject variability.

    Parameters
    ----------
    signals : np.ndarray, shape (n_samples, n_channels, n_times)
    subjects : np.ndarray, shape (n_samples,)
        Subject identifiers for each sample.
    reg : float
        Regularization for covariance estimation.

    Returns
    -------
    recentered : np.ndarray, same shape as signals
    """
    unique_subjects = np.unique(subjects)
    recentered = np.empty_like(signals)

    for subj in unique_subjects:
        mask = subjects == subj
        subj_signals = signals[mask]  # (n_trials, n_ch, n_times)
        n_trials = subj_signals.shape[0]
        n_ch = subj_signals.shape[1]

        log.info("Riemannian re-centering: subject %s (%d trials)", subj, n_trials)

        if n_trials < 2:
            log.warning("Subject %s has < 2 trials, skipping re-centering", subj)
            recentered[mask] = subj_signals
            continue

        # Compute per-trial covariance matrices
        covs = np.zeros((n_trials, n_ch, n_ch))
        for t in range(n_trials):
            x = subj_signals[t]  # (n_ch, n_times)
            # Remove mean per channel
            x_centered = x - x.mean(axis=1, keepdims=True)
            cov = (x_centered @ x_centered.T) / (x.shape[1] - 1)
            # Regularize
            cov += reg * np.eye(n_ch)
            covs[t] = cov

        # Compute geometric mean
        geom_mean = _geometric_mean_spd(covs)

        # Compute M^{-1/2}
        M_isqrt = _spd_sqrt_inv(geom_mean)

        # Whiten each trial: X_new = M^{-1/2} @ X
        for t in range(n_trials):
            recentered[mask][t] = M_isqrt @ subj_signals[t]

        # Verify assignment (numpy advanced indexing creates copies)
        # Need to use direct index assignment
        indices = np.where(mask)[0]
        for i, idx in enumerate(indices):
            recentered[idx] = M_isqrt @ subj_signals[i]

    return recentered


def euclidean_recenter(signals, subjects, reg=1e-6):
    """Apply Euclidean alignment: per-subject arithmetic mean whitening.

    Parameters
    ----------
    signals : np.ndarray, shape (n_samples, n_channels, n_times)
    subjects : np.ndarray, shape (n_samples,)
    reg : float
        Regularization.

    Returns
    -------
    recentered : np.ndarray, same shape as signals
    """
    unique_subjects = np.unique(subjects)
    recentered = np.empty_like(signals)

    for subj in unique_subjects:
        mask = subjects == subj
        subj_signals = signals[mask]
        n_trials = subj_signals.shape[0]
        n_ch = subj_signals.shape[1]

        log.info("Euclidean re-centering: subject %s (%d trials)", subj, n_trials)

        if n_trials < 2:
            log.warning("Subject %s has < 2 trials, skipping re-centering", subj)
            recentered[mask] = subj_signals
            continue

        # Compute arithmetic mean covariance
        mean_cov = np.zeros((n_ch, n_ch))
        for t in range(n_trials):
            x = subj_signals[t]
            x_centered = x - x.mean(axis=1, keepdims=True)
            mean_cov += (x_centered @ x_centered.T) / (x.shape[1] - 1)
        mean_cov /= n_trials
        mean_cov += reg * np.eye(n_ch)

        # Compute R^{-1/2}
        R_isqrt = _spd_sqrt_inv(mean_cov)

        indices = np.where(mask)[0]
        for i, idx in enumerate(indices):
            recentered[idx] = R_isqrt @ subj_signals[i]

    return recentered


def minmax_scale(signals):
    """Apply per-sample min-max normalization to [-1, 1].

    Matches BENDR's preprocessing normalization.

    Parameters
    ----------
    signals : np.ndarray, shape (n_samples, n_channels, n_times)

    Returns
    -------
    scaled : np.ndarray, same shape
    """
    # Per-sample min/max across channels and time
    mins = signals.min(axis=(1, 2), keepdims=True)
    maxs = signals.max(axis=(1, 2), keepdims=True)
    ranges = maxs - mins
    # Avoid division by zero
    ranges[ranges == 0] = 1.0
    return 2.0 * (signals - mins) / ranges - 1.0


# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------

def save_to_hdf5(
    filepath,
    signals,
    labels,
    subjects,
    dataset_name,
    sfreq,
    target_ch_names,
    method="spline",
    alpha=1e-5,
):
    """Save interpolated EEG to HDF5."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("signals", data=signals.astype(np.float32), compression="gzip")
        f.create_dataset("labels", data=labels.astype(np.int64))

        if subjects.dtype.kind in ("U", "S", "O"):
            f.create_dataset("subjects", data=np.array([s.encode("utf-8") if isinstance(s, str) else str(s).encode("utf-8") for s in subjects]), dtype=h5py.string_dtype())
        else:
            f.create_dataset("subjects", data=subjects)

        # Store channel names
        f.create_dataset(
            "channel_names",
            data=np.array(target_ch_names, dtype="S10"),
        )

        # Metadata
        f.attrs["dataset_name"] = dataset_name
        f.attrs["method"] = f"interpolate_{method}"
        f.attrs["n_channels"] = len(target_ch_names)
        f.attrs["sfreq"] = sfreq
        f.attrs["n_samples"] = signals.shape[0]
        f.attrs["n_times"] = signals.shape[2]
        f.attrs["alpha"] = alpha

    log.info(
        "Saved %s: %d samples, shape %s",
        filepath, signals.shape[0], signals.shape,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_SFREQ = {
    "bcic2a": 200.0,
    "physionet": 200.0,
    "tuev": 200.0,
    "faced": 200.0,
    "isruc-sleep": 200.0,
    "mdd_mumtaz2016": 200.0,
}


def process_dataset(dataset_name, output_dir, method, alpha, tuev_path,
                    recenter=None, bandpass=None, normalization="minmax"):
    """Process a single dataset end-to-end."""
    log.info("=" * 60)
    log.info("Processing dataset: %s (method=%s, alpha=%s, recenter=%s, bandpass=%s)",
             dataset_name, method, alpha, recenter, bandpass)
    log.info("=" * 60)

    # Load raw epochs
    if dataset_name == "tuev":
        epochs_list, labels, subjects = load_tuev(tuev_path)
    elif dataset_name == "bcic2a":
        epochs_list, labels, subjects = load_bcic2a()
    elif dataset_name == "physionet":
        epochs_list, labels, subjects = load_physionet()
    elif dataset_name == "faced":
        epochs_list, labels, subjects = load_faced()
    elif dataset_name == "isruc-sleep":
        epochs_list, labels, subjects = load_isruc_sleep()
    elif dataset_name == "mdd_mumtaz2016":
        epochs_list, labels, subjects = load_mdd_mumtaz()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    log.info("Loaded %d epoch batches, %d total labels", len(epochs_list), len(labels))

    if len(epochs_list) == 0:
        log.error("No data loaded for %s", dataset_name)
        return

    # Apply spherical spline interpolation to BENDR's 19 channels
    log.info("Applying spherical spline interpolation to %d target channels...",
             len(BENDR_19_CHANNELS))
    signals = apply_interpolation(epochs_list, BENDR_19_CHANNELS, alpha=alpha)

    # Verify shape
    assert signals.shape[0] == len(labels), (
        f"Mismatch: {signals.shape[0]} signals vs {len(labels)} labels"
    )
    assert signals.shape[1] == len(BENDR_19_CHANNELS), (
        f"Unexpected n_channels: {signals.shape[1]} (expected {len(BENDR_19_CHANNELS)})"
    )

    log.info("Interpolated signals shape: %s", signals.shape)

    # Apply bandpass filter if requested
    if bandpass is not None:
        lo, hi = bandpass
        log.info("Applying bandpass filter: %.1f - %.1f Hz", lo, hi)
        sfreq = DATASET_SFREQ[dataset_name]
        # Use MNE's filter on the raw signals
        for i in range(signals.shape[0]):
            signals[i] = mne.filter.filter_data(
                signals[i], sfreq, l_freq=lo, h_freq=hi, verbose=False
            )
        log.info("Bandpass filter applied")

    # Apply Riemannian or Euclidean re-centering if requested
    if recenter == "riemannian":
        log.info("Applying Riemannian re-centering...")
        signals = riemannian_recenter(signals, subjects)
        log.info("Riemannian re-centering done")
    elif recenter == "euclidean":
        log.info("Applying Euclidean re-centering...")
        signals = euclidean_recenter(signals, subjects)
        log.info("Euclidean re-centering done")

    # Apply normalization
    if normalization == "minmax":
        signals = minmax_scale(signals)
        log.info("Applied min-max normalization to [-1, 1]")
    else:
        log.info("Skipping normalization (raw signals)")

    # Build output filename
    suffix = method
    if normalization == "none":
        suffix += "_raw"
    if recenter:
        suffix += f"_recenter_{recenter}"
    if bandpass:
        suffix += f"_bp{int(bandpass[0])}-{int(bandpass[1])}"

    output_path = Path(output_dir) / f"{dataset_name}_interpolated_{suffix}.h5"
    save_to_hdf5(
        output_path,
        signals,
        labels,
        subjects,
        dataset_name,
        DATASET_SFREQ[dataset_name],
        BENDR_19_CHANNELS,
        method=method,
        alpha=alpha,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute MNE spherical spline interpolated EEG for BENDR"
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
        "--method",
        type=str,
        default="spline",
        choices=["spline", "MNE"],
        help="Interpolation method (default: spline)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="Regularization parameter for spherical spline (default: 1e-5)",
    )
    parser.add_argument(
        "--tuev-path",
        type=str,
        default=DEFAULT_TUEV_PATH,
        help="Path to TUEV dataset directory",
    )
    parser.add_argument(
        "--recenter",
        type=str,
        default=None,
        choices=["riemannian", "euclidean"],
        help="Re-centering method: riemannian (geometric mean) or euclidean (arithmetic mean)",
    )
    parser.add_argument(
        "--bandpass",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Bandpass filter frequencies in Hz (e.g., --bandpass 8 32)",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["minmax", "none"],
        help="Normalization: minmax [-1,1] (default) or none (raw)",
    )

    args = parser.parse_args()

    log.info("MNE Interpolation Preprocessing")
    log.info("Output dir: %s", args.output_dir)
    log.info("Method: %s (alpha=%s)", args.method, args.alpha)
    log.info("Re-centering: %s", args.recenter or "none")
    log.info("Bandpass: %s", args.bandpass or "none")
    log.info("Target channels: %s", BENDR_19_CHANNELS)
    log.info("Datasets: %s", args.dataset)

    for dataset_name in args.dataset:
        process_dataset(
            dataset_name, args.output_dir, args.method, args.alpha, args.tuev_path,
            recenter=args.recenter, bandpass=args.bandpass,
            normalization=args.normalization,
        )

    log.info("All datasets processed!")


if __name__ == "__main__":
    main()
