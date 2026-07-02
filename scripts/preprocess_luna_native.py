#!/usr/bin/env python
"""
Save raw EEG with native channels + 3D electrode positions as HDF5 for LUNA.

Unlike the interpolation preprocessing, this does NOT change channels.
Each dataset keeps its original channel count. Data is resampled to 128 Hz
to match LUNA's pretraining frequency.

Output format: HDF5 files with:
  - signals: [n_samples, n_chans, n_times] (native channel count)
  - labels: [n_samples]
  - subjects: [n_samples]
  - channel_locations: [n_chans, 3] (3D positions from standard_1020)
  - channel_names: [n_chans] (channel name strings)

Usage:
    python scripts/preprocess_luna_native.py --dataset bcic2a physionet tuev
"""

import argparse
import logging
from pathlib import Path

import h5py
import mne
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/luna_native")
DEFAULT_TUEV_PATH = "/expanse/projects/nemar/eeg_finetuning/data/tuh_eeg/tuh_eeg_events"

LUNA_SFREQ = 128.0  # LUNA's pretraining frequency

CHANNEL_RENAME_MAP = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "A1": "M1", "A2": "M2",
}


# ---------------------------------------------------------------------------
# Dataset loaders (reused from preprocess_interpolate.py)
# ---------------------------------------------------------------------------

def load_bcic2a():
    from moabb.datasets import BNCI2014_001
    dataset = BNCI2014_001()
    subjects = dataset.subject_list

    all_epochs, all_labels, all_subjects = [], [], []
    event_id = {"feet": 0, "left_hand": 1, "right_hand": 2, "tongue": 3}

    for subj in subjects:
        log.info("Loading BCI2a subject %d", subj)
        data = dataset.get_data(subjects=[subj])
        for session_name, session_data in data[subj].items():
            for run_name, raw in session_data.items():
                raw = raw.copy().pick("eeg")
                if raw.info["sfreq"] != LUNA_SFREQ:
                    raw.resample(LUNA_SFREQ)
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
                    raw, events, event_id=mapped_event_id,
                    tmin=0, tmax=4.0 - 1.0 / raw.info["sfreq"],
                    baseline=None, preload=True,
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
    from moabb.datasets import PhysionetMI
    dataset = PhysionetMI()
    subjects = dataset.subject_list

    all_epochs, all_labels, all_subjects = [], [], []
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
                    continue
                raw.pick(available)

                if raw.info["sfreq"] != LUNA_SFREQ:
                    raw.resample(LUNA_SFREQ)
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
                    raw, events, event_id=mapped_event_id,
                    tmin=0, tmax=3.0 - 1.0 / raw.info["sfreq"],
                    baseline=None, preload=True,
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


def load_tuev(tuev_path):
    from collections import Counter

    ch_names = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "A1", "A2", "Fz", "Cz", "Pz",
    ]
    event_id = {"spsw": 0, "gped": 1, "pled": 2, "eyem": 3, "artf": 4, "bckg": 5}
    rec_label_map = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}

    all_epochs, all_labels, all_subjects = [], [], []

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

                # Rename TUEV channels: "EEG FP1-REF" -> "Fp1"
                rename = {}
                for ch in raw.ch_names:
                    clean = ch.replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()
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

                if raw.info["sfreq"] != LUNA_SFREQ:
                    raw.resample(LUNA_SFREQ)
                raw.filter(l_freq=0.1, h_freq=None)

                # Parse .rec annotations
                rec_path = edf_path.with_suffix(".rec")
                if not rec_path.exists():
                    continue

                time_labels = {}
                with open(rec_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
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

                annotations = [(s, e, l) for s, e, l in segments if l in event_id]

                sfreq = raw.info["sfreq"]
                window_sec = 5.0
                for onset, offset, label in annotations:
                    duration = offset - onset
                    if duration < window_sec:
                        continue
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
                        epoch = mne.EpochsArray(data[np.newaxis, :, :], info, verbose=False)
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
# Channel location extraction
# ---------------------------------------------------------------------------

def get_channel_locations(ch_names_list):
    """Get 3D positions from standard_1020 montage for a list of channel names."""
    montage = mne.channels.make_standard_montage("standard_1020")
    all_positions = montage.get_positions()["ch_pos"]

    # Apply rename map to find positions for old-style names
    positions = []
    valid_names = []
    for ch in ch_names_list:
        # Try direct lookup
        lookup_name = ch
        for old, new in CHANNEL_RENAME_MAP.items():
            if ch == old:
                lookup_name = new
                break

        if lookup_name in all_positions:
            positions.append(all_positions[lookup_name])
            valid_names.append(ch)
        else:
            log.warning("Channel %s (lookup: %s) not in standard_1020, using zeros", ch, lookup_name)
            positions.append(np.zeros(3))
            valid_names.append(ch)

    return np.array(positions, dtype=np.float32), valid_names


def normalize_epoch_channel_names(epochs):
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
    if epochs.get_montage() is None:
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            epochs.set_montage(montage, on_missing="warn")
        except Exception as e:
            log.warning("Could not set montage: %s", e)
    return epochs


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def extract_signals(epochs_list):
    """Extract numpy arrays from epoch list, truncating to min time length."""
    all_signals = []
    for epochs in epochs_list:
        data = epochs.get_data()  # [n_epochs, n_chans, n_times]
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        all_signals.append(data)

    if not all_signals:
        raise RuntimeError("No signals extracted")

    # Truncate to min time length
    min_t = min(s.shape[2] for s in all_signals)
    all_signals = [s[:, :, :min_t] for s in all_signals]
    return np.concatenate(all_signals, axis=0)


def minmax_scale(signals):
    mins = signals.min(axis=(1, 2), keepdims=True)
    maxs = signals.max(axis=(1, 2), keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return 2.0 * (signals - mins) / ranges - 1.0


def save_to_hdf5(filepath, signals, labels, subjects, ch_names_list, ch_locations,
                 dataset_name, sfreq):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("signals", data=signals.astype(np.float32), compression="gzip")
        f.create_dataset("labels", data=labels.astype(np.int64))

        if subjects.dtype.kind in ("U", "S", "O"):
            f.create_dataset("subjects", data=[str(s).encode("utf-8") for s in subjects], dtype=h5py.string_dtype())
        else:
            f.create_dataset("subjects", data=subjects)

        f.create_dataset("channel_locations", data=ch_locations.astype(np.float32))
        f.create_dataset("channel_names", data=np.array(ch_names_list, dtype="S10"))

        f.attrs["dataset_name"] = dataset_name
        f.attrs["method"] = "native"
        f.attrs["n_channels"] = signals.shape[1]
        f.attrs["sfreq"] = sfreq
        f.attrs["n_samples"] = signals.shape[0]
        f.attrs["n_times"] = signals.shape[2]

    log.info("Saved %s: %d samples, shape %s", filepath, signals.shape[0], signals.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_dataset(dataset_name, output_dir, tuev_path, normalization="minmax"):
    log.info("=" * 60)
    log.info("Processing dataset: %s (native mode, sfreq=%d Hz)", dataset_name, int(LUNA_SFREQ))
    log.info("=" * 60)

    if dataset_name == "tuev":
        epochs_list, labels, subjects = load_tuev(tuev_path)
    elif dataset_name == "bcic2a":
        epochs_list, labels, subjects = load_bcic2a()
    elif dataset_name == "physionet":
        epochs_list, labels, subjects = load_physionet()
    elif dataset_name == "faced":
        epochs_list, labels, subjects = load_faced(target_sfreq=LUNA_SFREQ)
    elif dataset_name == "isruc-sleep":
        epochs_list, labels, subjects = load_isruc_sleep(target_sfreq=LUNA_SFREQ)
    elif dataset_name == "mdd_mumtaz2016":
        epochs_list, labels, subjects = load_mdd_mumtaz(target_sfreq=LUNA_SFREQ)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    log.info("Loaded %d epoch batches, %d total labels", len(epochs_list), len(labels))

    if len(epochs_list) == 0:
        log.error("No data loaded for %s", dataset_name)
        return

    # Normalize channel names and set montage
    epochs_list = [normalize_epoch_channel_names(ep) for ep in epochs_list]
    epochs_list = [ensure_montage(ep) for ep in epochs_list]

    # Get channel names and locations from first epoch
    ch_names_list = list(epochs_list[0].ch_names)
    ch_locations, _ = get_channel_locations(ch_names_list)
    log.info("Channels: %d, names: %s", len(ch_names_list), ch_names_list)
    log.info("Channel locations shape: %s", ch_locations.shape)

    # Extract signals
    signals = extract_signals(epochs_list)
    log.info("Signals shape: %s", signals.shape)

    assert signals.shape[0] == len(labels)

    # Normalize
    if normalization == "minmax":
        signals = minmax_scale(signals)
        log.info("Applied min-max normalization to [-1, 1]")
    else:
        log.info("Skipping normalization (raw signals)")

    # Save
    norm_suffix = "_raw" if normalization == "none" else ""
    output_path = Path(output_dir) / f"{dataset_name}_native_128hz{norm_suffix}.h5"
    save_to_hdf5(
        output_path, signals, labels, subjects,
        ch_names_list, ch_locations, dataset_name, LUNA_SFREQ,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Save raw EEG with native channels for LUNA experiments"
    )
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=["bcic2a", "physionet", "tuev", "faced", "isruc-sleep", "mdd_mumtaz2016"],
                        choices=["bcic2a", "physionet", "tuev", "faced", "isruc-sleep", "mdd_mumtaz2016"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tuev-path", type=str, default=DEFAULT_TUEV_PATH)

    parser.add_argument("--normalization", type=str, default="minmax", choices=["minmax", "none"])
    args = parser.parse_args()

    log.info("LUNA Native Preprocessing (sfreq=%d Hz)", int(LUNA_SFREQ))
    log.info("Output dir: %s", args.output_dir)
    log.info("Datasets: %s", args.dataset)

    for dataset_name in args.dataset:
        process_dataset(dataset_name, args.output_dir, args.tuev_path, args.normalization)

    log.info("All datasets processed!")


if __name__ == "__main__":
    main()
