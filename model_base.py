"""
model_base.py
MSTGNet Revision Core Library
Journal: Knowledge-Based Systems
Manuscript ID: KNOSYS-D-26-02299

Revision-ready design:
- Full MSTGNet default: use_cross_modal=False
- Cross-modal attention: ablation variant only
- Active features: 1,503 via raw_to_active_indices
- Scaling: fit StandardScaler on train fold only, transform val/test
- Early stopping: validation AUC
- Threshold: Youden's J from validation fold
- Prediction output: video-level CSV with probability, prediction, threshold
- Resume-safe outputs for fold results and prediction CSVs
- Traditional ML baselines: LogReg, RF, SVM-RBF, GradientBoosting with temporal-stat features
- Optional mean-only traditional ML features for reproducing older paper baseline
- Deep learning baselines: LSTM, BiLSTM, GRU, CNN-LSTM, Transformer
- Matched Transformer uses positional encoding
- Cross-dataset temporal-stat CORAL uses unlabeled target train+val distribution, not target test
- CORAL default is diagonal CORAL to avoid full covariance memory blow-up on 13,527-D temporal-stat features
- Cross-dataset MSTGNet source-only is supported
- Checkpoint selection helpers are validation-only, never test-based
- Interpretability uses positive-logit gradient and writes to interpretability/
- Gradient stability aggregation across folds/checkpoints is supported
"""

# ============================================================
# Imports
# ============================================================

import os
import gc
import json
import time
import math
import random
import pickle
import platform
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable, Iterable, Union

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

try:
    from scipy.stats import wilcoxon, ttest_rel
    from scipy.stats import t as scipy_t
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    scipy_t = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


warnings.simplefilter("default")


# ============================================================
# Constants
# ============================================================

DEFAULT_OUTPUT_ROOT = "outputs/revision_final"
DEFAULT_PROCESSED_DIR = "data/processed"

REQUIRED_PRED_COLS = [
    "dataset",
    "model",
    "seed",
    "repeat",
    "fold",
    "split",
    "video_id",
    "y_true",
    "y_prob",
    "y_pred",
    "threshold",
    "threshold_source",
    "source_dataset",
    "target_dataset",
]

PRED_KEY_COLS = [
    "dataset",
    "model",
    "seed",
    "repeat",
    "fold",
    "split",
    "video_id",
]

FOLD_KEY_COLS = [
    "dataset",
    "model",
    "seed",
    "repeat",
    "fold",
]

DEFAULT_METRIC_COLS = [
    "test_auc",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_mcc",
    "test_f1",
]


# ============================================================
# Reproducibility and utilities
# ============================================================

def set_all_seeds(seed: int = 42, deterministic: bool = True) -> Dict[str, Any]:
    """Set seeds for Python, NumPy, and PyTorch."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    deterministic_ok = True
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            deterministic_ok = False
            warnings.warn(
                "torch.use_deterministic_algorithms(True) failed. "
                "Small nondeterministic variation may occur.",
                RuntimeWarning,
            )

    return {
        "seed": seed,
        "cuda_available": bool(cuda_available),
        "deterministic_requested": bool(deterministic),
        "deterministic_algorithms": bool(deterministic_ok),
    }


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_dump(obj: Any, path: Union[str, Path]) -> None:
    """JSON dump that safely handles NumPy, Path, and Torch objects."""
    path = Path(path)
    ensure_dir(path.parent)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=_default,
            allow_nan=True,
        )


def safe_json_load(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_total_parameters(model: nn.Module) -> int:
    """Count all parameters."""
    return int(sum(p.numel() for p in model.parameters()))


def create_experiment_manifest(
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    *,
    run_name: str = "revision_final_v5",
) -> Dict[str, Any]:
    """Create reproducibility manifest."""
    return {
        "run_name": run_name,
        "created_at": now_string(),
        "output_dir": str(output_dir),
        "config": config,
        "revision_protocol": {
            "full_model_use_cross_modal": False,
            "cross_modal_attention_role": "ablation_variant_only",
            "threshold_strategy": "validation_youden",
            "early_stopping_metric": "validation_auc",
            "feature_protocol": "1503 active features via raw_to_active_indices",
            "scaling": "StandardScaler fit on train fold only",
            "prediction_level": "video-level probability after sequence probability averaging",
            "classification_head": "single binary logit trained with BCEWithLogitsLoss; sigmoid used for probability",
            "transformer_baseline": "matched temporal self-attention with positional encoding",
            "coral_protocol": "temporal-stat CORAL uses unlabeled target train+validation distribution for adaptation; target test is held out",
            "coral_default": "diagonal CORAL for high-dimensional temporal-stat features",
            "checkpoint_selection": "validation-only; never test_auc",
            "interpretability": "positive-logit input-gradient saliency",
            "gradient_stability": "top-k frequency aggregation across checkpoints/folds",
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def save_experiment_manifest(manifest: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir / "config")
    safe_json_dump(manifest, output_dir / "config" / "experiment_manifest.json")


def append_or_update_csv(
    csv_path: Union[str, Path],
    row_or_df: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
    key_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Resume-safe CSV writer."""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    if isinstance(row_or_df, dict):
        new_df = pd.DataFrame([row_or_df])
    elif isinstance(row_or_df, list):
        new_df = pd.DataFrame(row_or_df)
    else:
        new_df = row_or_df.copy()

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    if key_cols:
        if all(c in out_df.columns for c in key_cols):
            out_df = out_df.drop_duplicates(subset=key_cols, keep="last")
        else:
            missing = [c for c in key_cols if c not in out_df.columns]
            warnings.warn(
                f"Not deduplicating {csv_path} because key columns are missing: {missing}",
                RuntimeWarning,
            )

    out_df.to_csv(csv_path, index=False)
    return out_df


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _format_float(x: float, ndigits: int = 4) -> str:
    try:
        x = float(x)
        if np.isfinite(x):
            return f"{x:.{ndigits}f}"
        return "nan"
    except Exception:
        return "nan"


# ============================================================
# Data loading and sequence preparation
# ============================================================

def load_processed_dataset(
    dataset_name: str,
    processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
    expected_observed_features: Optional[int] = 1533,
    expected_active_features: Optional[int] = 1503,
) -> Dict[str, Any]:
    """Load revision-ready processed pickle."""
    path = Path(processed_dir) / f"mstgnet_{dataset_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    print(f"Loading processed dataset: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    required = [
        "video_to_frames",
        "video_ids",
        "video_labels",
        "coord_cols_observed",
        "coord_cols_active",
        "active_modality_slices",
        "raw_to_active_indices",
        "feature_mapping",
        "metadata",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in {path}: {missing}")

    n_observed = len(data["coord_cols_observed"])
    n_active = len(data["coord_cols_active"])

    if expected_observed_features is not None and n_observed != int(expected_observed_features):
        warnings.warn(
            f"Observed features={n_observed}, expected={expected_observed_features}. "
            "Continuing because dataset-driven metadata is authoritative.",
            UserWarning,
        )

    if expected_active_features is not None and n_active != int(expected_active_features):
        warnings.warn(
            f"Active features={n_active}, expected={expected_active_features}. "
            "Continuing because dataset-driven metadata is authoritative.",
            UserWarning,
        )

    raw_to_active = list(map(int, data["raw_to_active_indices"]))
    if len(raw_to_active) != n_active:
        raise AssertionError("raw_to_active_indices length does not match coord_cols_active length.")

    slices = data["active_modality_slices"]
    if "body" in slices and int(slices["body"][1]) != n_active:
        raise AssertionError("active_modality_slices do not end at active feature count.")

    video_to_frames = data["video_to_frames"]
    if len(video_to_frames) == 0:
        raise ValueError("video_to_frames is empty.")

    sample_vids = list(video_to_frames.keys())[:5]
    for vid in sample_vids:
        arr = video_to_frames[vid]["features"]
        if arr.shape[1] != n_observed:
            raise AssertionError(
                f"Video {vid} feature dimension={arr.shape[1]}, expected observed={n_observed}"
            )
        if not np.isfinite(arr).all():
            raise AssertionError(f"Non-finite values found in video {vid}")

    print(f"✓ {dataset_name} loaded")
    print(f"  Videos: {len(data['video_ids'])}")
    print(f"  Observed features: {n_observed}")
    print(f"  Active features:   {n_active}")
    print(f"  Active slices:     {data['active_modality_slices']}")
    return data


def get_video_labels_from_data(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return video_ids and video-level labels in dataset order."""
    video_ids = np.asarray(data["video_ids"]).astype(str)
    video_to_frames = data["video_to_frames"]

    labels = []
    for vid in video_ids:
        vid = str(vid)
        if vid in video_to_frames and "label" in video_to_frames[vid]:
            labels.append(int(video_to_frames[vid]["label"]))
        elif "video_labels" in data:
            labels_map = data["video_labels"]
            if isinstance(labels_map, dict):
                labels.append(int(labels_map[vid]))
            else:
                labels.append(int(np.asarray(labels_map)[len(labels)]))
        else:
            raise KeyError(f"Cannot infer label for video {vid}")

    return video_ids, np.asarray(labels, dtype=int)


def prepare_sequences(
    data: Dict[str, Any],
    seq_len: int = 50,
    stride: int = 25,
    *,
    use_active_features: bool = True,
    pad_short_videos: bool = True,
    pad_mode: str = "edge",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert video-level frame arrays into sliding-window sequences.
    All split decisions must be made at video level downstream.
    """
    video_to_frames = data["video_to_frames"]
    video_ids = list(data["video_ids"])
    raw_to_active = np.asarray(data["raw_to_active_indices"], dtype=np.int64)

    X_list, y_list, vid_list = [], [], []

    for vid in video_ids:
        vid_str = str(vid)
        if vid_str not in video_to_frames:
            warnings.warn(f"Video ID {vid_str} not found in video_to_frames; skipped.", UserWarning)
            continue

        item = video_to_frames[vid_str]
        frames = np.asarray(item["features"], dtype=np.float32)
        label = int(item["label"])

        if use_active_features:
            frames = frames[:, raw_to_active]

        n_frames = int(frames.shape[0])
        if n_frames <= 0:
            warnings.warn(f"Video {vid_str} has zero frames; skipped.", UserWarning)
            continue

        if n_frames < seq_len:
            if not pad_short_videos:
                continue
            pad_len = int(seq_len - n_frames)
            if pad_mode == "edge":
                pad_values = np.repeat(frames[-1:, :], pad_len, axis=0)
            elif pad_mode == "zero":
                pad_values = np.zeros((pad_len, frames.shape[1]), dtype=frames.dtype)
            else:
                raise ValueError("pad_mode must be 'edge' or 'zero'")
            seq = np.concatenate([frames, pad_values], axis=0)
            X_list.append(seq.astype(np.float32))
            y_list.append(label)
            vid_list.append(vid_str)
        else:
            for start in range(0, n_frames - seq_len + 1, stride):
                seq = frames[start:start + seq_len]
                X_list.append(seq.astype(np.float32))
                y_list.append(label)
                vid_list.append(vid_str)

    if len(X_list) == 0:
        raise ValueError("No sequences were created. Check seq_len, stride, and input data.")

    X_seq = np.stack(X_list).astype(np.float32)
    y_seq = np.asarray(y_list, dtype=np.int64)
    video_seq_ids = np.asarray(vid_list).astype(str)

    if verbose:
        print("Prepared sequences:")
        print(f"  X_seq: {X_seq.shape}")
        print(f"  y_seq: {y_seq.shape}")
        print(f"  video_seq_ids: {video_seq_ids.shape}")
        print(f"  unique videos: {len(np.unique(video_seq_ids))}")
        print(f"  class balance seq: {np.bincount(y_seq)}")

    return X_seq, y_seq, video_seq_ids


def make_video_level_splits(
    video_ids: np.ndarray,
    y_video: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    val_ratio: float = 0.28,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Make stratified video-level train/val/test splits."""
    video_ids = np.asarray(video_ids).astype(str)
    y_video = np.asarray(y_video).astype(int)

    if len(video_ids) != len(y_video):
        raise ValueError("video_ids and y_video length mismatch.")

    splits = []
    for repeat in range(int(n_repeats)):
        repeat_seed = int(seed) + repeat
        skf = StratifiedKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=repeat_seed,
        )

        for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(video_ids, y_video), start=1):
            trainval_ids = video_ids[trainval_idx]
            trainval_y = y_video[trainval_idx]
            val_split_seed = repeat_seed + fold_idx

            try:
                train_ids, val_ids, _, _ = train_test_split(
                    trainval_ids,
                    trainval_y,
                    test_size=float(val_ratio),
                    random_state=val_split_seed,
                    stratify=trainval_y,
                )
            except ValueError:
                warnings.warn(
                    f"Stratified validation split failed for repeat={repeat+1}, fold={fold_idx}; "
                    "falling back to non-stratified split.",
                    RuntimeWarning,
                )
                train_ids, val_ids, _, _ = train_test_split(
                    trainval_ids,
                    trainval_y,
                    test_size=float(val_ratio),
                    random_state=val_split_seed,
                    stratify=None,
                )

            splits.append(
                {
                    "repeat": int(repeat + 1),
                    "fold": int(fold_idx),
                    "seed": int(seed),
                    "split_seed": int(repeat_seed),
                    "val_split_seed": int(val_split_seed),
                    "train_videos": [str(v) for v in train_ids],
                    "val_videos": [str(v) for v in val_ids],
                    "test_videos": [str(v) for v in video_ids[test_idx]],
                }
            )

    print(f"Created {len(splits)} splits: {n_splits} folds × {n_repeats} repeats")
    return splits


def save_splits(splits: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    safe_json_dump({"splits": splits}, path)


def load_splits(path: Union[str, Path]) -> List[Dict[str, Any]]:
    obj = safe_json_load(path)
    return obj["splits"]


def fit_train_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler only on training sequences."""
    if len(X_train) == 0:
        raise ValueError("Cannot fit scaler on empty X_train.")
    scaler = StandardScaler()
    flat = np.asarray(X_train, dtype=np.float32).reshape(-1, X_train.shape[-1])
    scaler.fit(flat)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply scaler to sequences."""
    if len(X) == 0:
        raise ValueError("Cannot apply scaler to empty X.")
    n, t, f = X.shape
    flat = np.asarray(X, dtype=np.float32).reshape(-1, f)
    scaled = scaler.transform(flat).reshape(n, t, f)
    return scaled.astype(np.float32)


def rebuild_scaler_from_checkpoint(ckpt: Dict[str, Any]) -> StandardScaler:
    """Rebuild StandardScaler from checkpoint fields."""
    if "scaler_mean" not in ckpt or "scaler_scale" not in ckpt:
        raise KeyError("Checkpoint must contain 'scaler_mean' and 'scaler_scale'.")
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = int(len(scaler.mean_))
    return scaler


def split_sequence_data_by_video(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    split: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Select train/val/test sequences using video IDs."""
    train_set = set(map(str, split["train_videos"]))
    val_set = set(map(str, split["val_videos"]))
    test_set = set(map(str, split["test_videos"]))

    vids = np.asarray([str(v) for v in video_seq_ids])
    train_mask = np.array([v in train_set for v in vids])
    val_mask = np.array([v in val_set for v in vids])
    test_mask = np.array([v in test_set for v in vids])

    return {
        "train": (X_seq[train_mask], y_seq[train_mask], vids[train_mask]),
        "val": (X_seq[val_mask], y_seq[val_mask], vids[val_mask]),
        "test": (X_seq[test_mask], y_seq[test_mask], vids[test_mask]),
    }


# ============================================================
# Leakage audit helpers
# ============================================================

def audit_video_split_overlap(splits: List[Dict[str, Any]]) -> pd.DataFrame:
    """Audit whether train/val/test video IDs overlap within each split."""
    rows = []
    for s in splits:
        train = set(map(str, s.get("train_videos", [])))
        val = set(map(str, s.get("val_videos", [])))
        test = set(map(str, s.get("test_videos", [])))
        rows.append(
            {
                "repeat": int(s.get("repeat", -1)),
                "fold": int(s.get("fold", -1)),
                "n_train_videos": len(train),
                "n_val_videos": len(val),
                "n_test_videos": len(test),
                "overlap_train_val": len(train & val),
                "overlap_train_test": len(train & test),
                "overlap_val_test": len(val & test),
                "status": "PASS" if len(train & val) == 0 and len(train & test) == 0 and len(val & test) == 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def _extract_subject_mapping_if_available(data: Dict[str, Any]) -> Dict[str, str]:
    """Best-effort subject mapping extraction."""
    candidate_keys = [
        "subject_id",
        "participant_id",
        "person_id",
        "speaker_id",
        "actor_id",
        "trial_subject",
    ]
    mapping = {}

    for container_key in ["video_metadata", "metadata_by_video", "video_info"]:
        obj = data.get(container_key, None)
        if isinstance(obj, dict):
            for vid, meta in obj.items():
                if isinstance(meta, dict):
                    for ck in candidate_keys:
                        if ck in meta:
                            mapping[str(vid)] = str(meta[ck])
                            break

    meta = data.get("metadata", None)

    if isinstance(meta, pd.DataFrame):
        vid_col = next((c for c in ["video_id", "video", "id"] if c in meta.columns), None)
        subj_col = next((c for c in candidate_keys if c in meta.columns), None)
        if vid_col is not None and subj_col is not None:
            for _, r in meta.iterrows():
                mapping[str(r[vid_col])] = str(r[subj_col])

    if isinstance(meta, dict):
        for container_key in ["video_metadata", "metadata_by_video", "video_info"]:
            obj = meta.get(container_key, None)
            if isinstance(obj, dict):
                for vid, m in obj.items():
                    if isinstance(m, dict):
                        for ck in candidate_keys:
                            if ck in m:
                                mapping[str(vid)] = str(m[ck])
                                break

    return mapping


def audit_subject_split_overlap_if_available(
    data: Dict[str, Any],
    splits: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Audit subject-level overlap if subject IDs are available."""
    subject_map = _extract_subject_mapping_if_available(data)
    rows = []

    if not subject_map:
        for s in splits:
            rows.append(
                {
                    "repeat": int(s.get("repeat", -1)),
                    "fold": int(s.get("fold", -1)),
                    "status": "SUBJECT_ID_UNAVAILABLE",
                    "overlap_train_val_subjects": np.nan,
                    "overlap_train_test_subjects": np.nan,
                    "overlap_val_test_subjects": np.nan,
                }
            )
        return pd.DataFrame(rows)

    for s in splits:
        def subj_set(video_list):
            return {subject_map[str(v)] for v in video_list if str(v) in subject_map}

        train_subj = subj_set(s.get("train_videos", []))
        val_subj = subj_set(s.get("val_videos", []))
        test_subj = subj_set(s.get("test_videos", []))

        rows.append(
            {
                "repeat": int(s.get("repeat", -1)),
                "fold": int(s.get("fold", -1)),
                "n_train_subjects": len(train_subj),
                "n_val_subjects": len(val_subj),
                "n_test_subjects": len(test_subj),
                "overlap_train_val_subjects": len(train_subj & val_subj),
                "overlap_train_test_subjects": len(train_subj & test_subj),
                "overlap_val_test_subjects": len(val_subj & test_subj),
                "status": "PASS_NO_SUBJECT_OVERLAP"
                if len(train_subj & val_subj) == 0 and len(train_subj & test_subj) == 0 and len(val_subj & test_subj) == 0
                else "SUBJECT_OVERLAP_PRESENT",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Dataset and DataLoader
# ============================================================

class MSTGNetDataset(Dataset):
    """Dataset without full tensor copy."""

    def __init__(self, X: np.ndarray, y: np.ndarray, video_ids: np.ndarray):
        self.X = X
        self.y = np.asarray(y, dtype=np.float32)
        self.video_ids = np.asarray(video_ids).astype(str)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        vid = self.video_ids[idx]
        return x, y, vid


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build deterministic DataLoader."""
    ds = MSTGNetDataset(X, y, video_ids)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=worker_init_fn if int(num_workers) > 0 else None,
    )


# ============================================================
# Model components
# ============================================================

class GraphConstructionLayer(nn.Module):
    """Attention-derived temporal graph layer."""

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(q.size(-1)))
        adj = torch.softmax(scores, dim=-1)
        adj = self.dropout(adj)
        out = torch.matmul(adj, v)
        out = self.out(out)
        return self.norm(x + self.dropout(out))


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale temporal block with local and dilated longer-local branches."""

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        if int(hidden_dim) % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4.")

        branch_dim = int(hidden_dim) // 4
        self.conv_k3 = nn.Conv1d(hidden_dim, branch_dim, kernel_size=3, padding=1, dilation=1)
        self.conv_k5 = nn.Conv1d(hidden_dim, branch_dim, kernel_size=5, padding=2, dilation=1)
        self.conv_dilated = nn.Conv1d(hidden_dim, branch_dim, kernel_size=3, padding=2, dilation=2)
        self.conv_pointwise = nn.Conv1d(hidden_dim, branch_dim, kernel_size=1, padding=0)
        self.proj = nn.Conv1d(branch_dim * 4, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = x.transpose(1, 2)
        out = torch.cat(
            [
                F.relu(self.conv_k3(z)),
                F.relu(self.conv_k5(z)),
                F.relu(self.conv_dilated(z)),
                F.relu(self.conv_pointwise(z)),
            ],
            dim=1,
        )
        out = self.proj(out).transpose(1, 2)
        return self.norm(residual + self.dropout(out))


class AdaptiveTemporalPooling(nn.Module):
    """Attention pooling over time."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        hidden_dim = int(hidden_dim)
        mid = max(hidden_dim // 2, 1)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.Tanh(),
            nn.Linear(mid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding computed dynamically for sequence length."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, h = x.shape
        if h != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {h}")

        position = torch.arange(t, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, h, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / max(h, 1))
        )
        pe = torch.zeros(t, h, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        if h > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return self.dropout(x + pe.unsqueeze(0))


class CrossModalAttention(nn.Module):
    """Cross-modal attention over modality embeddings; ablation only."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, modality_embeddings: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(modality_embeddings, modality_embeddings, modality_embeddings)
        return self.norm(modality_embeddings + self.dropout(out))


class ModalityEncoder(nn.Module):
    """Per-modality encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        use_graph: bool = True,
        use_temporal: bool = True,
        use_pooling: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_graph = bool(use_graph)
        self.use_temporal = bool(use_temporal)
        self.use_pooling = bool(use_pooling)

        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.graph = GraphConstructionLayer(int(hidden_dim), dropout=float(dropout))
        self.temporal = MultiScaleTemporalBlock(int(hidden_dim), dropout=float(dropout))
        self.pool = AdaptiveTemporalPooling(int(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        if self.use_graph:
            h = self.graph(h)
        if self.use_temporal:
            h = self.temporal(h)
        if self.use_pooling:
            return self.pool(h)
        return h.mean(dim=1)


class MSTGNet(nn.Module):
    """Revision-ready MSTGNet. Default: use_cross_modal=False."""

    def __init__(
        self,
        input_dim: int,
        modality_slices: Dict[str, List[int]],
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
        enabled_modalities: Tuple[str, ...] = ("face", "eyes", "body"),
        use_graph: bool = True,
        use_temporal: bool = True,
        use_pooling: bool = True,
        use_cross_modal: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.modality_slices = modality_slices
        self.enabled_modalities = tuple(enabled_modalities)
        self.hidden_dim = int(hidden_dim)
        self.use_cross_modal = bool(use_cross_modal)

        if len(self.enabled_modalities) == 0:
            raise ValueError("enabled_modalities must not be empty.")

        max_end = max(int(modality_slices[m][1]) for m in self.enabled_modalities)
        if max_end > self.input_dim:
            raise ValueError(f"Modality slices exceed input_dim: max_end={max_end}, input_dim={self.input_dim}")

        encoders = {}
        for mod in self.enabled_modalities:
            if mod not in modality_slices:
                raise KeyError(f"Modality '{mod}' not found in modality_slices.")
            start, end = modality_slices[mod]
            mod_dim = int(end) - int(start)
            if mod_dim <= 0:
                raise ValueError(f"Invalid modality dimension for {mod}: {mod_dim}")
            encoders[mod] = ModalityEncoder(
                mod_dim,
                self.hidden_dim,
                use_graph=use_graph,
                use_temporal=use_temporal,
                use_pooling=use_pooling,
                dropout=dropout,
            )

        self.encoders = nn.ModuleDict(encoders)
        self.cross_modal = (
            CrossModalAttention(self.hidden_dim, num_heads=num_heads, dropout=dropout)
            if self.use_cross_modal else None
        )

        classifier_in = self.hidden_dim * len(self.enabled_modalities)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for mod in self.enabled_modalities:
            start, end = self.modality_slices[mod]
            x_mod = x[:, :, int(start):int(end)]
            embeddings.append(self.encoders[mod](x_mod))

        z = torch.stack(embeddings, dim=1)
        if self.cross_modal is not None:
            z = self.cross_modal(z)
        return z.reshape(z.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x)).squeeze(-1)


class MatchedTemporalSelfAttention(nn.Module):
    """Matched Transformer/self-attention baseline with positional encoding."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.3,
        dim_feedforward_multiplier: int = 2,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.positional_encoding = (
            SinusoidalPositionalEncoding(int(hidden_dim), dropout=float(dropout))
            if use_positional_encoding else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * int(dim_feedforward_multiplier),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.pool = AdaptiveTemporalPooling(int(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.LayerNorm(int(hidden_dim) // 2),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.positional_encoding(h)
        h = self.encoder(h)
        return self.pool(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x)).squeeze(-1)


class TemporalRNNClassifier(nn.Module):
    """LSTM/BiLSTM/GRU baseline for raw sequence input."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        num_layers: int = 1,
        dropout: float = 0.3,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        rnn_type = str(rnn_type).lower()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be 'lstm' or 'gru'.")

        self.rnn_type = rnn_type
        self.bidirectional = bool(bidirectional)
        self.use_attention_pooling = bool(use_attention_pooling)

        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=int(hidden_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )

        out_dim = int(hidden_dim) * (2 if bidirectional else 1)
        self.pool = AdaptiveTemporalPooling(out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.LayerNorm(int(hidden_dim) // 2),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.rnn(h)
        if self.use_attention_pooling:
            return self.pool(out)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x)).squeeze(-1)


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM baseline for raw sequence input."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        conv_kernel_size: int = 5,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        padding = int(conv_kernel_size) // 2
        norm1 = nn.BatchNorm1d(int(hidden_dim)) if use_batchnorm else nn.Identity()
        norm2 = nn.BatchNorm1d(int(hidden_dim)) if use_batchnorm else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=int(conv_kernel_size), padding=padding),
            norm1,
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            norm2,
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        self.lstm = nn.LSTM(
            input_size=int(hidden_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(lstm_layers),
            batch_first=True,
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            bidirectional=False,
        )
        self.pool = AdaptiveTemporalPooling(int(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.LayerNorm(int(hidden_dim) // 2),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        z = h.transpose(1, 2)
        z = self.conv(z).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.pool(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x)).squeeze(-1)


# ============================================================
# Model factories
# ============================================================

def build_mstgnet_factory(
    *,
    input_dim: int,
    modality_slices: Dict[str, List[int]],
    hidden_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.3,
    enabled_modalities: Tuple[str, ...] = ("face", "eyes", "body"),
    use_graph: bool = True,
    use_temporal: bool = True,
    use_pooling: bool = True,
    use_cross_modal: bool = False,
) -> Callable[[], nn.Module]:
    return lambda: MSTGNet(
        input_dim=input_dim,
        modality_slices=modality_slices,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        enabled_modalities=enabled_modalities,
        use_graph=use_graph,
        use_temporal=use_temporal,
        use_pooling=use_pooling,
        use_cross_modal=use_cross_modal,
    )


def build_transformer_factory(
    *,
    input_dim: int,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    dropout: float = 0.3,
    dim_feedforward_multiplier: int = 2,
    use_positional_encoding: bool = True,
) -> Callable[[], nn.Module]:
    return lambda: MatchedTemporalSelfAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        dim_feedforward_multiplier=dim_feedforward_multiplier,
        use_positional_encoding=use_positional_encoding,
    )


def build_lstm_factory(
    *,
    input_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    num_layers: int = 1,
) -> Callable[[], nn.Module]:
    return lambda: TemporalRNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        rnn_type="lstm",
        bidirectional=False,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_bilstm_factory(
    *,
    input_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    num_layers: int = 1,
) -> Callable[[], nn.Module]:
    return lambda: TemporalRNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        rnn_type="lstm",
        bidirectional=True,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_gru_factory(
    *,
    input_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    num_layers: int = 1,
) -> Callable[[], nn.Module]:
    return lambda: TemporalRNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        rnn_type="gru",
        bidirectional=False,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_cnn_lstm_factory(
    *,
    input_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    conv_kernel_size: int = 5,
    lstm_layers: int = 1,
    use_batchnorm: bool = True,
) -> Callable[[], nn.Module]:
    return lambda: CNNLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        conv_kernel_size=conv_kernel_size,
        lstm_layers=lstm_layers,
        use_batchnorm=use_batchnorm,
    )


# ============================================================
# Metrics and thresholding
# ============================================================

def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return float("nan")


def choose_youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Choose threshold on validation set using Youden's J."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    idx = int(np.argmax(tpr - fpr))
    thr = float(thresholds[idx])
    if not np.isfinite(thr):
        thr = 0.5
    return float(np.clip(thr, 0.0, 1.0))


def evaluate_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate binary metrics at threshold."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty y_true.")

    y_pred = (probs >= float(threshold)).astype(int)

    auc = safe_auc(y_true, probs)
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    except Exception:
        tn = fp = fn = tp = 0
        sensitivity = specificity = float("nan")

    try:
        brier = float(brier_score_loss(y_true, probs))
    except Exception:
        brier = float("nan")

    return {
        "auc": auc,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": brier,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def aggregate_sequence_probs_to_video(
    y_seq: np.ndarray,
    probs_seq: np.ndarray,
    video_ids_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average sequence probabilities per video."""
    df = pd.DataFrame(
        {
            "video_id": [str(v) for v in video_ids_seq],
            "y": np.asarray(y_seq).astype(int),
            "prob": np.asarray(probs_seq).astype(float),
        }
    )

    label_nunique = df.groupby("video_id")["y"].nunique()
    bad = label_nunique[label_nunique > 1]
    if len(bad) > 0:
        raise AssertionError(f"Inconsistent sequence labels within videos: {bad.head().to_dict()}")

    agg = df.groupby("video_id", sort=True).agg(
        y_true=("y", "first"),
        y_prob=("prob", "mean"),
    ).reset_index()

    return (
        agg["video_id"].astype(str).to_numpy(),
        agg["y_true"].astype(int).to_numpy(),
        agg["y_prob"].astype(float).to_numpy(),
    )


# ============================================================
# Inference time utilities
# ============================================================

def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_warmup_batches: int = 2,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Measure ms per batch and ms per sequence."""
    model.eval()
    times = []
    total_sequences = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)

            if i < int(n_warmup_batches):
                _ = model(x)
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - t0
            times.append(elapsed)
            total_sequences += int(x.size(0))

            if max_batches is not None and len(times) >= int(max_batches):
                break

    if not times:
        return {"ms_per_batch": float("nan"), "ms_per_sequence": float("nan")}

    total_time = float(np.sum(times))
    return {
        "ms_per_batch": float(np.mean(times) * 1000.0),
        "ms_per_sequence": float((total_time / max(total_sequences, 1)) * 1000.0),
    }


def measure_inference_time_video(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_warmup_batches: int = 1,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Measure approximate ms per video using unique video IDs in loader."""
    model.eval()
    times = []
    seen_videos = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x, _y, vids = batch
            x = x.to(device, non_blocking=True)

            if i < int(n_warmup_batches):
                _ = model(x)
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - t0
            times.append(elapsed)
            seen_videos.extend([str(v) for v in vids])

            if max_batches is not None and len(times) >= int(max_batches):
                break

    if not times:
        return {
            "ms_per_video": float("nan"),
            "n_unique_videos_timed": 0,
            "total_inference_time_sec": float("nan"),
        }

    total_time = float(np.sum(times))
    n_unique_videos = int(len(set(seen_videos)))
    return {
        "ms_per_video": float((total_time / max(n_unique_videos, 1)) * 1000.0),
        "n_unique_videos_timed": n_unique_videos,
        "total_inference_time_sec": total_time,
    }


def _measure_sklearn_inference_time(
    clf: Any,
    X: np.ndarray,
    *,
    n_warmup: int = 1,
    n_repeats: int = 5,
) -> Dict[str, float]:
    """Measure approximate sklearn inference latency per video."""
    X = np.asarray(X)
    if len(X) == 0:
        return {"ms_per_video": float("nan"), "total_inference_time_sec": float("nan")}

    for _ in range(int(n_warmup)):
        _ = _safe_predict_proba_positive(clf, X)

    times = []
    for _ in range(int(n_repeats)):
        t0 = time.time()
        _ = _safe_predict_proba_positive(clf, X)
        times.append(time.time() - t0)

    mean_time = float(np.mean(times)) if times else float("nan")
    return {
        "ms_per_video": float((mean_time / max(len(X), 1)) * 1000.0),
        "total_inference_time_sec": mean_time,
    }


# ============================================================
# Training and prediction
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    gradient_clip: float = 5.0,
) -> float:
    model.train()
    losses = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip))
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_sequence_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_y, all_vids = [], [], []

    for x, y, vids in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())
        all_vids.extend([str(v) for v in vids])

    if len(all_probs) == 0:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(all_y).astype(int),
        np.concatenate(all_probs).astype(float),
        np.asarray(all_vids).astype(str),
    )


def predict_video_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_seq, probs_seq, video_ids_seq = predict_sequence_probs(model, loader, device)
    return aggregate_sequence_probs_to_video(y_seq, probs_seq, video_ids_seq)


def save_predictions(
    path: Union[str, Path],
    *,
    dataset: str,
    model_name: str,
    seed: int,
    repeat: int,
    fold: int,
    split: str,
    video_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    threshold_source: str = "validation_youden",
    source_dataset: Optional[str] = None,
    target_dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Save video-level predictions with required schema, resume-safe."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    df = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model_name,
            "seed": int(seed),
            "repeat": int(repeat),
            "fold": int(fold),
            "split": split,
            "video_id": [str(v) for v in video_ids],
            "y_true": np.asarray(y_true).astype(int),
            "y_prob": y_prob.astype(float),
            "y_pred": y_pred.astype(int),
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "source_dataset": source_dataset if source_dataset is not None else "",
            "target_dataset": target_dataset if target_dataset is not None else "",
        }
    )

    for col in REQUIRED_PRED_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[REQUIRED_PRED_COLS]

    path = Path(path)
    ensure_dir(path.parent)

    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
        out = out.drop_duplicates(subset=PRED_KEY_COLS, keep="last")
    else:
        out = df

    out.to_csv(path, index=False)
    return df


def save_fold_results(results: Dict[str, Any], path: Union[str, Path]) -> pd.DataFrame:
    return append_or_update_csv(path, results, key_cols=FOLD_KEY_COLS)


def validate_split_nonempty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    repeat: int,
    fold: int,
) -> None:
    """Fail fast for empty split or warn for single-class split."""
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Empty split detected: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}. "
            f"Check video ID consistency."
        )

    if len(np.unique(y_train)) < 2:
        warnings.warn(f"Train split has only one class: repeat={repeat}, fold={fold}", RuntimeWarning)
    if len(np.unique(y_val)) < 2:
        warnings.warn(f"Validation split has only one class: repeat={repeat}, fold={fold}", RuntimeWarning)
    if len(np.unique(y_test)) < 2:
        warnings.warn(f"Test split has only one class: repeat={repeat}, fold={fold}", RuntimeWarning)


def train_one_fold(
    *,
    dataset_name: str,
    model_name: str,
    model_factory: Callable[[], nn.Module],
    split: Dict[str, Any],
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    device: torch.device,
    output_dir: Union[str, Path],
    batch_size: int = 16,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    gradient_clip: float = 5.0,
    num_workers: int = 0,
    seed: int = 42,
    min_delta: float = 0.0,
    save_checkpoint: bool = True,
) -> Dict[str, Any]:
    """
    Train one fold with:
    - train-fold-only scaling
    - early stopping by validation AUC
    - Youden threshold from validation
    - test metrics at validation threshold
    """
    set_all_seeds(seed)

    output_dir = Path(output_dir)
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "fold_results")
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "logs")

    repeat = int(split["repeat"])
    fold = int(split["fold"])

    data_split = split_sequence_data_by_video(X_seq, y_seq, video_seq_ids, split)
    X_train, y_train, vid_train = data_split["train"]
    X_val, y_val, vid_val = data_split["val"]
    X_test, y_test, vid_test = data_split["test"]

    validate_split_nonempty(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        repeat=repeat,
        fold=fold,
    )

    scaler = fit_train_scaler(X_train)
    X_train = apply_scaler(scaler, X_train)
    X_val = apply_scaler(scaler, X_val)
    X_test = apply_scaler(scaler, X_test)

    train_loader = make_loader(
        X_train,
        y_train,
        vid_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        vid_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    test_loader = make_loader(
        X_test,
        y_test,
        vid_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    model = model_factory().to(device)
    n_params = count_parameters(model)
    n_total_params = count_total_parameters(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_val_auc = float("nan")
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    train_start = time.time()
    logs = []

    print(
        f"Fold setup: dataset={dataset_name}, model={model_name}, "
        f"repeat={repeat}, fold={fold}, seed={seed}, n_params={n_params}"
    )
    print(
        f"Sequences: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | "
        f"Videos: train={len(split['train_videos'])}, "
        f"val={len(split['val_videos'])}, test={len(split['test_videos'])}"
    )

    for epoch in range(1, int(epochs) + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_clip=gradient_clip,
        )

        val_vids, val_y_video, val_probs_video = predict_video_probs(
            model,
            val_loader,
            device,
        )
        val_auc = safe_auc(val_y_video, val_probs_video)

        logs.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": int(seed),
                "repeat": repeat,
                "fold": fold,
                "epoch": int(epoch),
                "train_loss": float(loss),
                "val_auc": float(val_auc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = False
        if best_state is None:
            improved = True
        elif np.isfinite(val_auc) and (
            (not np.isfinite(best_val_auc)) or val_auc > best_val_auc + float(min_delta)
        ):
            improved = True

        if improved:
            best_val_auc = float(val_auc) if np.isfinite(val_auc) else float("nan")
            best_epoch = int(epoch)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"  epoch={epoch:03d} | loss={_format_float(loss)} | "
                f"val_auc={_format_float(val_auc)} | "
                f"best_val_auc={_format_float(best_val_auc)} | "
                f"bad_epochs={bad_epochs}"
            )

        if bad_epochs >= int(patience):
            print(
                f"  Early stopping at epoch={epoch}. "
                f"Best epoch={best_epoch}, best val AUC={_format_float(best_val_auc)}"
            )
            break

    train_time_sec = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        warnings.warn("best_state is None. Using last epoch model.", RuntimeWarning)

    val_vids, val_y_video, val_probs_video = predict_video_probs(
        model,
        val_loader,
        device,
    )
    threshold = choose_youden_threshold(val_y_video, val_probs_video)
    val_metrics = evaluate_at_threshold(val_y_video, val_probs_video, threshold)

    test_vids, test_y_video, test_probs_video = predict_video_probs(
        model,
        test_loader,
        device,
    )
    test_metrics = evaluate_at_threshold(test_y_video, test_probs_video, threshold)

    inference_timing = measure_inference_time(
        model,
        test_loader,
        device,
        n_warmup_batches=1,
        max_batches=None,
    )
    inference_timing_video = measure_inference_time_video(
        model,
        test_loader,
        device,
        n_warmup_batches=1,
        max_batches=None,
    )

    pred_path = output_dir / "predictions" / f"{model_name}_{dataset_name}_video_predictions.csv"

    save_predictions(
        pred_path,
        dataset=dataset_name,
        model_name=model_name,
        seed=seed,
        repeat=repeat,
        fold=fold,
        split="val",
        video_ids=val_vids,
        y_true=val_y_video,
        y_prob=val_probs_video,
        threshold=threshold,
        threshold_source="validation_youden",
    )

    save_predictions(
        pred_path,
        dataset=dataset_name,
        model_name=model_name,
        seed=seed,
        repeat=repeat,
        fold=fold,
        split="test",
        video_ids=test_vids,
        y_true=test_y_video,
        y_prob=test_probs_video,
        threshold=threshold,
        threshold_source="validation_youden",
    )

    ckpt_path = output_dir / "checkpoints" / f"{model_name}_{dataset_name}_seed{seed}_rep{repeat}_fold{fold}.pt"

    if save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "threshold": float(threshold),
                "threshold_source": "validation_youden",
                "best_val_auc": float(best_val_auc),
                "best_epoch": int(best_epoch),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "split": split,
                "dataset": dataset_name,
                "model_name": model_name,
                "seed": int(seed),
                "repeat": repeat,
                "fold": fold,
                "n_params": int(n_params),
                "n_trainable_params": int(n_params),
                "n_total_params": int(n_total_params),
            },
            ckpt_path,
        )

    pd.DataFrame(logs).to_csv(
        output_dir / "logs" / f"{model_name}_{dataset_name}_seed{seed}_rep{repeat}_fold{fold}_trainlog.csv",
        index=False,
    )

    fold_result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": int(seed),
        "repeat": repeat,
        "fold": fold,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "last_train_loss": float(logs[-1]["train_loss"]) if len(logs) else float("nan"),
        "threshold": float(threshold),
        "threshold_source": "validation_youden",
        "train_time_sec": float(train_time_sec),
        "n_params": int(n_params),
        "n_trainable_params": int(n_params),
        "n_total_params": int(n_total_params),
        "n_train_seq": int(len(X_train)),
        "n_val_seq": int(len(X_val)),
        "n_test_seq": int(len(X_test)),
        "n_train_videos": int(len(split["train_videos"])),
        "n_val_videos": int(len(split["val_videos"])),
        "n_test_videos": int(len(split["test_videos"])),
        "ms_per_batch": float(inference_timing.get("ms_per_batch", float("nan"))),
        "ms_per_sequence": float(inference_timing.get("ms_per_sequence", float("nan"))),
        "ms_per_video": float(inference_timing_video.get("ms_per_video", float("nan"))),
        "checkpoint_path": str(ckpt_path) if save_checkpoint else "",
    }

    for k, v in val_metrics.items():
        fold_result[f"val_{k}"] = v

    for k, v in test_metrics.items():
        fold_result[f"test_{k}"] = v

    save_fold_results(
        fold_result,
        output_dir / "fold_results" / f"{model_name}_{dataset_name}.csv",
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return fold_result


def run_cv_experiment(
    *,
    dataset_name: str,
    model_name: str,
    model_factory: Callable[[], nn.Module],
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    splits: List[Dict[str, Any]],
    device: torch.device,
    output_dir: Union[str, Path],
    batch_size: int = 16,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    gradient_clip: float = 5.0,
    seed: int = 42,
    save_checkpoint: bool = True,
) -> pd.DataFrame:
    """Run CV experiment with fold-specific seeds and saved splits."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "config" / "splits")

    save_splits(
        splits,
        output_dir / "config" / "splits" / f"{dataset_name}_splits_seed{seed}.json",
    )

    results = []
    for split in splits:
        print("\n" + "=" * 70)
        print(f"Running {model_name} | {dataset_name} | repeat={split['repeat']} fold={split['fold']}")
        print("=" * 70)

        fold_seed = int(seed) + (int(split["repeat"]) - 1) * 100 + int(split["fold"])

        row = train_one_fold(
            dataset_name=dataset_name,
            model_name=model_name,
            model_factory=model_factory,
            split=split,
            X_seq=X_seq,
            y_seq=y_seq,
            video_seq_ids=video_seq_ids,
            device=device,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            seed=fold_seed,
            save_checkpoint=save_checkpoint,
        )
        results.append(row)

        pd.DataFrame(results).to_csv(
            output_dir / "tables" / f"{model_name}_{dataset_name}_cv_summary.csv",
            index=False,
        )

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "tables" / f"{model_name}_{dataset_name}_cv_summary.csv", index=False)
    return df


def filter_completed_splits(
    splits: List[Dict[str, Any]],
    *,
    fold_results_path: Union[str, Path],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Skip splits that already exist in fold_results CSV.

    Fold seed convention:
        fold_seed = seed + (repeat - 1) * 100 + fold
    """
    fold_results_path = Path(fold_results_path)
    if not fold_results_path.exists():
        return splits

    existing = pd.read_csv(fold_results_path)
    required = {"seed", "repeat", "fold"}

    if not required.issubset(set(existing.columns)):
        warnings.warn(
            f"Cannot filter completed splits because {fold_results_path} "
            f"does not contain columns {required}. Running all splits.",
            RuntimeWarning,
        )
        return splits

    done = set(
        zip(
            existing["seed"].astype(int),
            existing["repeat"].astype(int),
            existing["fold"].astype(int),
        )
    )

    splits_to_run = []
    for s in splits:
        repeat = int(s["repeat"])
        fold = int(s["fold"])
        fold_seed = int(seed) + (repeat - 1) * 100 + fold
        if (fold_seed, repeat, fold) not in done:
            splits_to_run.append(s)

    print(
        f"filter_completed_splits: total={len(splits)}, "
        f"done={len(splits) - len(splits_to_run)}, "
        f"remaining={len(splits_to_run)}"
    )
    return splits_to_run


def select_checkpoint_validation_only(
    fold_results_path: Union[str, Path],
    *,
    seed_whitelist: Optional[Iterable[int]] = None,
    repeat_whitelist: Optional[Iterable[int]] = None,
    fold_whitelist: Optional[Iterable[int]] = None,
    checkpoint_col: str = "checkpoint_path",
) -> Optional[Path]:
    """
    Select checkpoint using validation metrics only.

    IMPORTANT:
    - Never uses test_auc.
    - Preferred metric: best_val_auc, then val_auc.
    """
    path = Path(fold_results_path)
    if not path.exists():
        warnings.warn(f"Fold results file not found: {path}", RuntimeWarning)
        return None

    df = pd.read_csv(path)

    if checkpoint_col not in df.columns:
        warnings.warn(f"No {checkpoint_col} column in {path}", RuntimeWarning)
        return None

    if seed_whitelist is not None and "seed" in df.columns:
        seeds = set(map(int, seed_whitelist))
        df = df[df["seed"].astype(int).isin(seeds)].copy()

    if repeat_whitelist is not None and "repeat" in df.columns:
        repeats = set(map(int, repeat_whitelist))
        df = df[df["repeat"].astype(int).isin(repeats)].copy()

    if fold_whitelist is not None and "fold" in df.columns:
        folds = set(map(int, fold_whitelist))
        df = df[df["fold"].astype(int).isin(folds)].copy()

    if len(df) == 0:
        warnings.warn(f"No rows left after checkpoint filters: {path}", RuntimeWarning)
        return None

    if "best_val_auc" in df.columns:
        df = df.sort_values("best_val_auc", ascending=False)
    elif "val_auc" in df.columns:
        df = df.sort_values("val_auc", ascending=False)
    else:
        warnings.warn(
            f"No validation metric found in {path}. Refusing to select checkpoint using test metrics.",
            RuntimeWarning,
        )
        return None

    for p in df[checkpoint_col].dropna().astype(str).tolist():
        if p and Path(p).exists():
            return Path(p)

    warnings.warn(f"No valid checkpoint path exists in {path}", RuntimeWarning)
    return None


def load_model_checkpoint_for_inference(
    *,
    checkpoint_path: Union[str, Path],
    model_factory: Callable[[], nn.Module],
    device: torch.device,
) -> Tuple[nn.Module, StandardScaler, float, Dict[str, Any]]:
    """
    Load trained model, rebuild scaler, and return validation-derived threshold.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = model_factory().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = rebuild_scaler_from_checkpoint(ckpt)
    threshold = float(ckpt.get("threshold", 0.5))
    return model, scaler, threshold, ckpt


# ============================================================
# Traditional ML feature extraction and baselines
# ============================================================

def extract_video_mean_features(
    data: Dict[str, Any],
    *,
    use_active_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract video-level mean features.

    This reproduces the older/simple traditional ML baseline:
    one 1,503-D vector per video when active features are used.
    """
    video_to_frames = data["video_to_frames"]
    video_ids = np.asarray(data["video_ids"]).astype(str)
    raw_to_active = np.asarray(data["raw_to_active_indices"], dtype=np.int64)

    X_rows, y_rows, used_vids = [], [], []

    for vid in video_ids:
        vid = str(vid)
        if vid not in video_to_frames:
            continue

        item = video_to_frames[vid]
        arr = np.asarray(item["features"], dtype=np.float32)
        if use_active_features:
            arr = arr[:, raw_to_active]

        X_rows.append(arr.mean(axis=0).astype(np.float32))
        y_rows.append(int(item["label"]))
        used_vids.append(vid)

    if len(X_rows) == 0:
        raise ValueError("No mean features extracted.")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows).astype(int)
    return X, y, np.asarray(used_vids).astype(str)


def extract_video_temporal_stat_features(
    data: Dict[str, Any],
    *,
    use_active_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract video-level temporal-statistical features:
    mean, std, min, max, range, velocity mean/std, acceleration mean/std.

    With 1,503 active features, output dimensionality is 13,527.
    """
    video_to_frames = data["video_to_frames"]
    video_ids = np.asarray(data["video_ids"]).astype(str)
    raw_to_active = np.asarray(data["raw_to_active_indices"], dtype=np.int64)

    X_rows, y_rows, used_vids = [], [], []

    for vid in video_ids:
        vid = str(vid)
        if vid not in video_to_frames:
            continue

        item = video_to_frames[vid]
        arr = np.asarray(item["features"], dtype=np.float32)
        if use_active_features:
            arr = arr[:, raw_to_active]

        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        minv = arr.min(axis=0)
        maxv = arr.max(axis=0)
        rangev = maxv - minv

        vel = np.diff(arr, axis=0)
        if len(vel) == 0:
            vel = np.zeros_like(arr[:1])
        vel_mean = vel.mean(axis=0)
        vel_std = vel.std(axis=0)

        acc = np.diff(vel, axis=0)
        if len(acc) == 0:
            acc = np.zeros_like(arr[:1])
        acc_mean = acc.mean(axis=0)
        acc_std = acc.std(axis=0)

        feats = np.concatenate(
            [
                mean,
                std,
                minv,
                maxv,
                rangev,
                vel_mean,
                vel_std,
                acc_mean,
                acc_std,
            ]
        ).astype(np.float32)

        X_rows.append(feats)
        y_rows.append(int(item["label"]))
        used_vids.append(vid)

    if len(X_rows) == 0:
        raise ValueError("No temporal-stat features extracted.")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows).astype(int)
    return X, y, np.asarray(used_vids).astype(str)


def extract_video_features_for_ml(
    data: Dict[str, Any],
    *,
    feature_mode: str = "temporal_stat",
    use_active_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract traditional ML features.

    feature_mode:
    - "temporal_stat": 9-stat descriptors, stronger revised baseline.
    - "mean": older 1,503-D mean descriptor for reproduction.
    """
    feature_mode = str(feature_mode).lower()
    if feature_mode in {"temporal_stat", "temporal-stat", "stat", "stats"}:
        return extract_video_temporal_stat_features(data, use_active_features=use_active_features)
    if feature_mode in {"mean", "mean_only", "mean-only"}:
        return extract_video_mean_features(data, use_active_features=use_active_features)
    raise ValueError("feature_mode must be 'temporal_stat' or 'mean'.")


def default_traditional_ml_models(seed: int = 42) -> Dict[str, Any]:
    """Default complete traditional ML baselines for revision."""
    return {
        "LogReg_temporal_stat": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=int(seed),
        ),
        "RF_temporal_stat": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        ),
        "SVM_temporal_stat": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=int(seed),
        ),
        "GradBoost_temporal_stat": GradientBoostingClassifier(
            random_state=int(seed),
        ),
    }


def _clone_and_set_random_state(estimator: Any, seed: int) -> Any:
    clf = clone(estimator)
    if hasattr(clf, "random_state"):
        try:
            clf.set_params(random_state=int(seed))
        except Exception:
            pass
    return clf


def _safe_predict_proba_positive(clf: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for sklearn classifier."""
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(float)
        return np.asarray(p).reshape(-1).astype(float)

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        return (1.0 / (1.0 + np.exp(-z))).astype(float)

    pred = clf.predict(X)
    return np.asarray(pred, dtype=float)


def run_traditional_ml_baseline(
    *,
    dataset_name: str,
    data: Dict[str, Any],
    splits: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    models: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    feature_mode: str = "temporal_stat",
) -> pd.DataFrame:
    """
    Run complete traditional ML baselines using identical video splits.

    feature_mode:
    - "temporal_stat" for reviewer-revised stronger baseline.
    - "mean" for older/simple 1,503-D baseline reproduction.
    """
    if models is None:
        models = default_traditional_ml_models(seed=seed)

    output_dir = Path(output_dir)
    ensure_dir(output_dir / "fold_results")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "tables")

    X_video, y_video, video_ids = extract_video_features_for_ml(
        data,
        feature_mode=feature_mode,
        use_active_features=True,
    )
    vid_to_idx = {str(v): i for i, v in enumerate(video_ids)}

    all_rows = []

    for model_name, clf_template in models.items():
        model_name_effective = model_name
        if feature_mode.lower().startswith("mean"):
            model_name_effective = model_name.replace("_temporal_stat", "_mean")

        for split in splits:
            repeat = int(split["repeat"])
            fold = int(split["fold"])
            fold_seed = int(seed) + (repeat - 1) * 100 + fold

            train_idx = [vid_to_idx[str(v)] for v in split["train_videos"]]
            val_idx = [vid_to_idx[str(v)] for v in split["val_videos"]]
            test_idx = [vid_to_idx[str(v)] for v in split["test_videos"]]

            X_train, y_train = X_video[train_idx], y_video[train_idx]
            X_val, y_val = X_video[val_idx], y_video[val_idx]
            X_test, y_test = X_video[test_idx], y_video[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            clf = _clone_and_set_random_state(clf_template, fold_seed)

            train_start = time.time()
            clf.fit(X_train_s, y_train)
            train_time_sec = time.time() - train_start

            val_prob = _safe_predict_proba_positive(clf, X_val_s)
            test_prob = _safe_predict_proba_positive(clf, X_test_s)

            threshold = choose_youden_threshold(y_val, val_prob)
            val_metrics = evaluate_at_threshold(y_val, val_prob, threshold)
            test_metrics = evaluate_at_threshold(y_test, test_prob, threshold)
            timing = _measure_sklearn_inference_time(clf, X_test_s)

            row = {
                "dataset": dataset_name,
                "model": model_name_effective,
                "seed": fold_seed,
                "repeat": repeat,
                "fold": fold,
                "feature_mode": feature_mode,
                "feature_dim": int(X_video.shape[1]),
                "threshold": float(threshold),
                "threshold_source": "validation_youden",
                "train_time_sec": float(train_time_sec),
                "n_params": np.nan,
                "n_trainable_params": np.nan,
                "n_total_params": np.nan,
                "n_train_videos": int(len(train_idx)),
                "n_val_videos": int(len(val_idx)),
                "n_test_videos": int(len(test_idx)),
                "n_train_seq": np.nan,
                "n_val_seq": np.nan,
                "n_test_seq": np.nan,
                "ms_per_batch": np.nan,
                "ms_per_sequence": np.nan,
                "ms_per_video": float(timing.get("ms_per_video", float("nan"))),
            }
            for k, v in val_metrics.items():
                row[f"val_{k}"] = v
            for k, v in test_metrics.items():
                row[f"test_{k}"] = v

            all_rows.append(row)
            save_fold_results(
                row,
                output_dir / "fold_results" / f"{model_name_effective}_{dataset_name}.csv",
            )

            pred_path = output_dir / "predictions" / f"{model_name_effective}_{dataset_name}_video_predictions.csv"

            save_predictions(
                pred_path,
                dataset=dataset_name,
                model_name=model_name_effective,
                seed=fold_seed,
                repeat=repeat,
                fold=fold,
                split="val",
                video_ids=video_ids[val_idx],
                y_true=y_val,
                y_prob=val_prob,
                threshold=threshold,
            )
            save_predictions(
                pred_path,
                dataset=dataset_name,
                model_name=model_name_effective,
                seed=fold_seed,
                repeat=repeat,
                fold=fold,
                split="test",
                video_ids=video_ids[test_idx],
                y_true=y_test,
                y_prob=test_prob,
                threshold=threshold,
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(
        output_dir / "tables" / f"traditional_ml_{feature_mode}_{dataset_name}_summary.csv",
        index=False,
    )
    return df

# ============================================================
# Deep learning baseline runner
# ============================================================

def build_deep_learning_baseline_factories(
    *,
    input_dim: int = 1503,
    hidden_dim: int = 128,
    num_heads: int = 4,
    transformer_num_layers: int = 4,
    dropout: float = 0.3,
    cnn_lstm_use_batchnorm: bool = True,
) -> Dict[str, Callable[[], nn.Module]]:
    """
    Build complete DL baseline factories:
    - LSTM
    - BiLSTM
    - GRU
    - CNN-LSTM
    - Transformer_L{layers}
    """
    return {
        "LSTM": build_lstm_factory(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=1,
        ),
        "BiLSTM": build_bilstm_factory(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=1,
        ),
        "GRU": build_gru_factory(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=1,
        ),
        "CNN-LSTM": build_cnn_lstm_factory(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            conv_kernel_size=5,
            lstm_layers=1,
            use_batchnorm=cnn_lstm_use_batchnorm,
        ),
        f"Transformer_L{int(transformer_num_layers)}": build_transformer_factory(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=int(transformer_num_layers),
            dropout=dropout,
            use_positional_encoding=True,
        ),
    }


def run_deep_learning_baselines(
    *,
    dataset_name: str,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    splits: List[Dict[str, Any]],
    device: torch.device,
    output_dir: Union[str, Path],
    input_dim: int = 1503,
    hidden_dim: int = 128,
    num_heads: int = 4,
    transformer_num_layers: int = 4,
    dropout: float = 0.3,
    cnn_lstm_use_batchnorm: bool = True,
    batch_size: int = 16,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    gradient_clip: float = 5.0,
    seed: int = 42,
    save_checkpoint: bool = True,
    model_whitelist: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Run DL baselines with the same CV, scaling, early-stopping, and threshold protocol.

    Caller should pass dataset-specific dropout explicitly, e.g. I3D=0.4, RLT=0.3.
    """
    output_dir = Path(output_dir)
    factories = build_deep_learning_baseline_factories(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        transformer_num_layers=transformer_num_layers,
        dropout=dropout,
        cnn_lstm_use_batchnorm=cnn_lstm_use_batchnorm,
    )

    if model_whitelist is not None:
        allowed = set(map(str, model_whitelist))
        factories = {k: v for k, v in factories.items() if k in allowed}

    all_dfs = []

    for model_name, factory in factories.items():
        fold_results_path = output_dir / "fold_results" / f"{model_name}_{dataset_name}.csv"
        splits_to_run = filter_completed_splits(
            splits,
            fold_results_path=fold_results_path,
            seed=seed,
        )

        if len(splits_to_run) == 0:
            print(f"{model_name} | {dataset_name}: all splits already complete.")
            existing = pd.read_csv(fold_results_path) if fold_results_path.exists() else pd.DataFrame()
            all_dfs.append(existing)
            continue

        df = run_cv_experiment(
            dataset_name=dataset_name,
            model_name=model_name,
            model_factory=factory,
            X_seq=X_seq,
            y_seq=y_seq,
            video_seq_ids=video_seq_ids,
            splits=splits_to_run,
            device=device,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            seed=seed,
            save_checkpoint=save_checkpoint,
        )
        all_dfs.append(df)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ============================================================
# Statistical tests and aggregation
# ============================================================

def _ci_from_values(vals: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Return SEM, CI low, CI high for 1D numeric values.
    Uses t critical when scipy is available and n > 1.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(len(vals))

    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(vals))

    if n == 1:
        return 0.0, mean, mean

    std = float(np.std(vals, ddof=1))
    sem = float(std / math.sqrt(n))

    if SCIPY_AVAILABLE and scipy_t is not None:
        alpha = 1.0 - float(ci)
        tcrit = float(scipy_t.ppf(1.0 - alpha / 2.0, n - 1))
    else:
        tcrit = 1.96

    low = mean - tcrit * sem
    high = mean + tcrit * sem
    return sem, float(low), float(high)


def _ci95_from_values(vals: np.ndarray) -> Tuple[float, float, float]:
    """Return SEM, CI95 low, CI95 high."""
    return _ci_from_values(vals, ci=0.95)


def summarize_cv_results(
    df: pd.DataFrame,
    *,
    metric_cols: Optional[List[str]] = None,
    group_cols: Optional[List[str]] = None,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Summarize CV results with mean, std, sem, CI, min, max, median, n.
    """
    if metric_cols is None:
        metric_cols = DEFAULT_METRIC_COLS
    if group_cols is None:
        group_cols = ["dataset", "model"]

    if df is None or len(df) == 0:
        return pd.DataFrame()

    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        warnings.warn(
            f"summarize_cv_results: missing group columns {missing_group_cols}. Returning empty DataFrame.",
            RuntimeWarning,
        )
        return pd.DataFrame()

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}

        for m in metric_cols:
            if m not in g.columns:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_sem"] = float("nan")
                row[f"{m}_ci{int(ci * 100)}_low"] = float("nan")
                row[f"{m}_ci{int(ci * 100)}_high"] = float("nan")
                row[f"{m}_median"] = float("nan")
                row[f"{m}_min"] = float("nan")
                row[f"{m}_max"] = float("nan")
                row[f"{m}_n"] = 0
                continue

            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(len(vals))

            if n == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_sem"] = float("nan")
                row[f"{m}_ci{int(ci * 100)}_low"] = float("nan")
                row[f"{m}_ci{int(ci * 100)}_high"] = float("nan")
                row[f"{m}_median"] = float("nan")
                row[f"{m}_min"] = float("nan")
                row[f"{m}_max"] = float("nan")
                row[f"{m}_n"] = 0
                continue

            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem, ci_low, ci_high = _ci_from_values(vals, ci=ci)

            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_sem"] = sem
            row[f"{m}_ci{int(ci * 100)}_low"] = ci_low
            row[f"{m}_ci{int(ci * 100)}_high"] = ci_high
            row[f"{m}_median"] = float(np.median(vals))
            row[f"{m}_min"] = float(np.min(vals))
            row[f"{m}_max"] = float(np.max(vals))
            row[f"{m}_n"] = n

            # Backward-compatible aliases for 95% CI.
            if abs(float(ci) - 0.95) < 1e-12:
                row[f"{m}_ci95_low"] = ci_low
                row[f"{m}_ci95_high"] = ci_high

        rows.append(row)

    return pd.DataFrame(rows)


def compute_repeat_level_ci(
    df: pd.DataFrame,
    *,
    metric: str = "test_auc",
    group_cols: Optional[List[str]] = None,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Compute repeat-level CI.

    This is more conservative for repeated CV because fold scores within a repeat
    may be correlated.
    """
    if group_cols is None:
        group_cols = ["dataset", "model"]

    if df is None or len(df) == 0 or metric not in df.columns:
        return pd.DataFrame()

    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        warnings.warn(
            f"compute_repeat_level_ci: missing group columns {missing_group_cols}. Returning empty DataFrame.",
            RuntimeWarning,
        )
        return pd.DataFrame()

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        if "repeat" not in g.columns:
            repeat_means = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
        else:
            repeat_means = (
                g.groupby("repeat")[metric]
                .mean(numeric_only=True)
                .dropna()
                .to_numpy(dtype=float)
            )

        repeat_means = repeat_means[np.isfinite(repeat_means)]
        n = int(len(repeat_means))
        mean = float(np.mean(repeat_means)) if n else float("nan")
        std = float(np.std(repeat_means, ddof=1)) if n > 1 else 0.0
        sem, low, high = _ci_from_values(repeat_means, ci=ci)

        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "metric": metric,
                "repeat_n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_level": float(ci),
                "aggregation_level": "repeat",
                "min": float(np.min(repeat_means)) if n else float("nan"),
                "max": float(np.max(repeat_means)) if n else float("nan"),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def wilcoxon_signed_rank_test(
    df: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    metric: str = "test_auc",
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test on fold-level metrics."""
    if not SCIPY_AVAILABLE:
        return {
            "test": "wilcoxon_signed_rank",
            "available": False,
            "reason": "scipy unavailable",
            "p_value": float("nan"),
        }

    if df is None or len(df) == 0:
        return {
            "test": "wilcoxon_signed_rank",
            "available": True,
            "reason": "empty dataframe",
            "p_value": float("nan"),
        }

    d = df.copy()
    if dataset is not None:
        d = d[d["dataset"].astype(str) == str(dataset)]

    key_cols = ["dataset", "seed", "repeat", "fold"]

    missing = [c for c in key_cols + ["model", metric] if c not in d.columns]
    if missing:
        return {
            "test": "wilcoxon_signed_rank",
            "available": True,
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "reason": f"missing columns: {missing}",
            "p_value": float("nan"),
        }

    a = d[d["model"].astype(str) == str(model_a)][key_cols + [metric]].rename(columns={metric: "a"})
    b = d[d["model"].astype(str) == str(model_b)][key_cols + [metric]].rename(columns={metric: "b"})
    merged = pd.merge(a, b, on=key_cols, how="inner")
    merged["a"] = pd.to_numeric(merged["a"], errors="coerce")
    merged["b"] = pd.to_numeric(merged["b"], errors="coerce")
    merged = merged.dropna(subset=["a", "b"])

    if len(merged) < 2:
        return {
            "test": "wilcoxon_signed_rank",
            "available": True,
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "n_pairs": int(len(merged)),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "reason": "fewer than 2 paired observations",
        }

    diff = merged["a"].to_numpy(dtype=float) - merged["b"].to_numpy(dtype=float)

    if np.allclose(diff, 0.0):
        return {
            "test": "wilcoxon_signed_rank",
            "available": True,
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "n_pairs": int(len(merged)),
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_a": float(merged["a"].mean()),
            "mean_b": float(merged["b"].mean()),
            "mean_diff_a_minus_b": 0.0,
            "median_diff_a_minus_b": 0.0,
            "note": "all paired differences are zero",
        }

    stat, p = wilcoxon(merged["a"].to_numpy(dtype=float), merged["b"].to_numpy(dtype=float))

    return {
        "test": "wilcoxon_signed_rank",
        "available": True,
        "dataset": dataset,
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n_pairs": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p),
        "mean_a": float(merged["a"].mean()),
        "mean_b": float(merged["b"].mean()),
        "mean_diff_a_minus_b": float(diff.mean()),
        "median_diff_a_minus_b": float(np.median(diff)),
    }


def paired_ttest(
    df: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    metric: str = "test_auc",
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Paired t-test on fold-level metrics."""
    if not SCIPY_AVAILABLE:
        return {
            "test": "paired_ttest",
            "available": False,
            "reason": "scipy unavailable",
            "p_value": float("nan"),
        }

    if df is None or len(df) == 0:
        return {
            "test": "paired_ttest",
            "available": True,
            "reason": "empty dataframe",
            "p_value": float("nan"),
        }

    d = df.copy()
    if dataset is not None:
        d = d[d["dataset"].astype(str) == str(dataset)]

    key_cols = ["dataset", "seed", "repeat", "fold"]

    missing = [c for c in key_cols + ["model", metric] if c not in d.columns]
    if missing:
        return {
            "test": "paired_ttest",
            "available": True,
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "reason": f"missing columns: {missing}",
            "p_value": float("nan"),
        }

    a = d[d["model"].astype(str) == str(model_a)][key_cols + [metric]].rename(columns={metric: "a"})
    b = d[d["model"].astype(str) == str(model_b)][key_cols + [metric]].rename(columns={metric: "b"})
    merged = pd.merge(a, b, on=key_cols, how="inner")
    merged["a"] = pd.to_numeric(merged["a"], errors="coerce")
    merged["b"] = pd.to_numeric(merged["b"], errors="coerce")
    merged = merged.dropna(subset=["a", "b"])

    if len(merged) < 2:
        return {
            "test": "paired_ttest",
            "available": True,
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "n_pairs": int(len(merged)),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "reason": "fewer than 2 paired observations",
        }

    stat, p = ttest_rel(
        merged["a"].to_numpy(dtype=float),
        merged["b"].to_numpy(dtype=float),
        nan_policy="omit",
    )

    diff = merged["a"].to_numpy(dtype=float) - merged["b"].to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]

    mean_diff = float(np.mean(diff)) if len(diff) else float("nan")
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    cohen_dz = float(mean_diff / std_diff) if len(diff) > 1 and std_diff > 0 else float("nan")

    return {
        "test": "paired_ttest",
        "available": True,
        "dataset": dataset,
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n_pairs": int(len(merged)),
        "statistic": float(stat),
        "p_value": float(p),
        "mean_a": float(merged["a"].mean()),
        "mean_b": float(merged["b"].mean()),
        "mean_diff_a_minus_b": mean_diff,
        "median_diff_a_minus_b": float(np.median(diff)) if len(diff) else float("nan"),
        "std_diff": std_diff,
        "cohen_dz": cohen_dz,
    }


def _holm_bonferroni(pvals: Iterable[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values.

    NaN p-values remain NaN.
    """
    pvals = np.asarray(list(pvals), dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    finite_idx = np.where(np.isfinite(pvals))[0]
    if len(finite_idx) == 0:
        return out.tolist()

    finite_p = pvals[finite_idx]
    order = np.argsort(finite_p)
    sorted_idx = finite_idx[order]
    sorted_p = finite_p[order]
    m = len(sorted_p)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for i, p in enumerate(sorted_p):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(running_max, 1.0)

    out[sorted_idx] = adjusted_sorted
    return out.tolist()


def compare_models_statistically(
    df: pd.DataFrame,
    *,
    reference_model: str = "MSTGNet",
    metric: str = "test_auc",
    datasets: Optional[Iterable[str]] = None,
    candidate_models: Optional[Iterable[str]] = None,
    adjust_pvalues: bool = True,
) -> pd.DataFrame:
    """
    Compare reference model against candidate models using paired Wilcoxon and paired t-test.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if "dataset" not in df.columns or "model" not in df.columns:
        raise KeyError("DataFrame must contain 'dataset' and 'model' columns.")

    if datasets is None:
        datasets = sorted(df["dataset"].dropna().astype(str).unique().tolist())

    if candidate_models is None:
        candidate_models = sorted(
            [m for m in df["model"].dropna().astype(str).unique().tolist() if m != reference_model]
        )

    rows = []

    for ds in datasets:
        for model_b in candidate_models:
            w = wilcoxon_signed_rank_test(
                df,
                model_a=reference_model,
                model_b=model_b,
                metric=metric,
                dataset=str(ds),
            )
            t = paired_ttest(
                df,
                model_a=reference_model,
                model_b=model_b,
                metric=metric,
                dataset=str(ds),
            )

            row = {
                "dataset": str(ds),
                "reference_model": reference_model,
                "comparison_model": model_b,
                "metric": metric,
                "wilcoxon_n_pairs": w.get("n_pairs", np.nan),
                "wilcoxon_statistic": w.get("statistic", np.nan),
                "wilcoxon_p_value": w.get("p_value", np.nan),
                "ttest_n_pairs": t.get("n_pairs", np.nan),
                "ttest_statistic": t.get("statistic", np.nan),
                "ttest_p_value": t.get("p_value", np.nan),
                "mean_reference": w.get("mean_a", t.get("mean_a", np.nan)),
                "mean_comparison": w.get("mean_b", t.get("mean_b", np.nan)),
                "mean_diff_reference_minus_comparison": w.get(
                    "mean_diff_a_minus_b",
                    t.get("mean_diff_a_minus_b", np.nan),
                ),
                "median_diff_reference_minus_comparison": w.get(
                    "median_diff_a_minus_b",
                    t.get("median_diff_a_minus_b", np.nan),
                ),
                "cohen_dz": t.get("cohen_dz", np.nan),
            }
            rows.append(row)

    out = pd.DataFrame(rows)

    if adjust_pvalues and len(out) > 0:
        out["wilcoxon_p_holm"] = np.nan
        out["ttest_p_holm"] = np.nan

        for ds, idx in out.groupby("dataset").groups.items():
            idx = list(idx)
            out.loc[idx, "wilcoxon_p_holm"] = _holm_bonferroni(out.loc[idx, "wilcoxon_p_value"].tolist())
            out.loc[idx, "ttest_p_holm"] = _holm_bonferroni(out.loc[idx, "ttest_p_value"].tolist())

    return out


# ============================================================
# Model capacity and computational summaries
# ============================================================

def compute_model_capacity_table(
    model_factories: Dict[str, Callable[[], nn.Module]],
    *,
    input_shape: Optional[Tuple[int, int, int]] = None,
    device: Optional[torch.device] = None,
    measure_forward_ms: bool = False,
    n_warmup: int = 3,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute model parameter counts and optional forward-pass latency.

    Parameters
    ----------
    model_factories:
        Dict mapping model name to zero-argument model factory.
    input_shape:
        Optional input shape [B, T, F] for forward latency measurement.
    device:
        Device for optional latency measurement.
    measure_forward_ms:
        If True, measure forward latency using synthetic input.
    """
    rows = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name, factory in model_factories.items():
        model = factory()
        n_trainable = count_parameters(model)
        n_total = count_total_parameters(model)

        row = {
            "model": model_name,
            "n_params": n_trainable,
            "n_trainable_params": n_trainable,
            "n_total_params": n_total,
            "param_millions": float(n_trainable / 1_000_000.0),
            "total_param_millions": float(n_total / 1_000_000.0),
            "device_for_timing": str(device) if measure_forward_ms else "",
            "forward_ms_per_batch": np.nan,
            "forward_ms_per_sample": np.nan,
        }

        if measure_forward_ms and input_shape is not None:
            model = model.to(device)
            model.eval()
            x = torch.randn(*input_shape, device=device)

            with torch.no_grad():
                for _ in range(int(n_warmup)):
                    _ = model(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                times = []
                for _ in range(int(n_repeats)):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.time() - t0)

            if times:
                ms_batch = float(np.mean(times) * 1000.0)
                row["forward_ms_per_batch"] = ms_batch
                row["forward_ms_per_sample"] = float(ms_batch / max(int(input_shape[0]), 1))

        rows.append(row)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)

    if len(out) > 0 and "MSTGNet" in out["model"].values:
        ref_params = float(out.loc[out["model"] == "MSTGNet", "n_trainable_params"].iloc[0])
        if ref_params > 0:
            out["param_ratio_vs_MSTGNet"] = out["n_trainable_params"].astype(float) / ref_params
        else:
            out["param_ratio_vs_MSTGNet"] = np.nan

    return out


# ============================================================
# Landmark quality analysis
# ============================================================

def _compute_video_quality_from_frames(
    arr: np.ndarray,
    *,
    modality_slices: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """
    Compute generic frame/landmark quality proxies for one video.

    These are proxy metrics:
    - finite/nonfinite rate
    - zero-value rate
    - frame-to-frame absolute velocity statistics
    They are not true MediaPipe tracking confidence unless the data explicitly stores confidence.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected arr with shape [T,F], got {arr.shape}")

    n_frames = int(arr.shape[0])
    n_features = int(arr.shape[1])

    finite_mask = np.isfinite(arr)
    finite_rate = float(np.mean(finite_mask)) if arr.size else float("nan")

    abs_arr = np.abs(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    zero_rate = float(np.mean(abs_arr == 0.0)) if arr.size else float("nan")

    if n_frames > 1:
        vel = np.diff(abs_arr, axis=0)
        mean_abs_velocity = float(np.mean(np.abs(vel)))
        std_abs_velocity = float(np.std(np.abs(vel)))
        p95_abs_velocity = float(np.percentile(np.abs(vel), 95))
    else:
        mean_abs_velocity = 0.0
        std_abs_velocity = 0.0
        p95_abs_velocity = 0.0

    row = {
        "n_frames": n_frames,
        "n_features": n_features,
        "finite_rate": finite_rate,
        "nonfinite_rate": float(1.0 - finite_rate) if np.isfinite(finite_rate) else float("nan"),
        "zero_rate": zero_rate,
        "mean_abs_value": float(np.mean(abs_arr)) if arr.size else float("nan"),
        "std_abs_value": float(np.std(abs_arr)) if arr.size else float("nan"),
        "mean_abs_velocity": mean_abs_velocity,
        "std_abs_velocity": std_abs_velocity,
        "p95_abs_velocity": p95_abs_velocity,
    }

    if modality_slices is not None:
        for mod, sl in modality_slices.items():
            start, end = int(sl[0]), int(sl[1])
            if start < 0 or end > n_features or end <= start:
                continue

            m = arr[:, start:end]
            m_finite = np.isfinite(m)
            m_abs = np.abs(np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0))

            row[f"{mod}_finite_rate"] = float(np.mean(m_finite)) if m.size else float("nan")
            row[f"{mod}_nonfinite_rate"] = (
                float(1.0 - row[f"{mod}_finite_rate"])
                if np.isfinite(row[f"{mod}_finite_rate"])
                else float("nan")
            )
            row[f"{mod}_zero_rate"] = float(np.mean(m_abs == 0.0)) if m.size else float("nan")
            row[f"{mod}_mean_abs_value"] = float(np.mean(m_abs)) if m.size else float("nan")
            row[f"{mod}_std_abs_value"] = float(np.std(m_abs)) if m.size else float("nan")

            if len(m_abs) > 1:
                mv = np.diff(m_abs, axis=0)
                row[f"{mod}_mean_abs_velocity"] = float(np.mean(np.abs(mv)))
                row[f"{mod}_std_abs_velocity"] = float(np.std(np.abs(mv)))
                row[f"{mod}_p95_abs_velocity"] = float(np.percentile(np.abs(mv), 95))
            else:
                row[f"{mod}_mean_abs_velocity"] = 0.0
                row[f"{mod}_std_abs_velocity"] = 0.0
                row[f"{mod}_p95_abs_velocity"] = 0.0

    return row


def compute_landmark_quality_table(
    data: Dict[str, Any],
    *,
    dataset_name: str = "",
    use_active_features: bool = True,
    long_format: bool = False,
) -> pd.DataFrame:
    """
    Compute per-video landmark quality proxies.

    Parameters
    ----------
    long_format:
        If False, returns one row per video with modality-prefixed columns.
        If True, returns one row per video per modality, easier for groupby(["dataset", "modality"]).
    """
    video_to_frames = data["video_to_frames"]
    video_ids = np.asarray(data["video_ids"]).astype(str)
    raw_to_active = np.asarray(data["raw_to_active_indices"], dtype=np.int64)

    if use_active_features:
        modality_slices = data.get("active_modality_slices", None)
    else:
        modality_slices = data.get("observed_modality_slices", None)

    rows = []

    for vid in video_ids:
        vid = str(vid)
        if vid not in video_to_frames:
            continue

        item = video_to_frames[vid]
        arr = np.asarray(item["features"], dtype=np.float32)

        if use_active_features:
            arr = arr[:, raw_to_active]

        base = {
            "dataset": dataset_name,
            "video_id": vid,
            "label": int(item.get("label", -1)),
            "use_active_features": bool(use_active_features),
        }

        if not long_format:
            row = dict(base)
            row.update(_compute_video_quality_from_frames(arr, modality_slices=modality_slices))
            rows.append(row)
        else:
            # Global/all-modality row.
            global_metrics = _compute_video_quality_from_frames(arr, modality_slices=None)
            row = dict(base)
            row["modality"] = "all"
            row.update(global_metrics)
            rows.append(row)

            if modality_slices is not None:
                n_features = int(arr.shape[1])
                for mod, sl in modality_slices.items():
                    start, end = int(sl[0]), int(sl[1])
                    if start < 0 or end > n_features or end <= start:
                        continue
                    m = arr[:, start:end]
                    mod_metrics = _compute_video_quality_from_frames(m, modality_slices=None)
                    row = dict(base)
                    row["modality"] = str(mod)
                    row.update(mod_metrics)
                    rows.append(row)

    return pd.DataFrame(rows)


def compute_landmark_quality_prediction_correlation(
    quality_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    quality_cols: Optional[List[str]] = None,
    prediction_split: str = "test",
    model_filter: Optional[str] = None,
    aggregate_predictions_per_video: bool = False,
) -> pd.DataFrame:
    """
    Correlate landmark quality proxies with prediction error.

    Error metrics:
    - abs_error_prob: absolute difference between y_true and y_prob
    - incorrect: binary incorrect prediction

    Notes
    -----
    If repeated CV predictions are included, each video may appear multiple times.
    Set aggregate_predictions_per_video=True for one independent row per video.
    """
    if quality_df is None or len(quality_df) == 0:
        return pd.DataFrame()
    if predictions_df is None or len(predictions_df) == 0:
        return pd.DataFrame()

    q = quality_df.copy()
    p = predictions_df.copy()

    q["video_id"] = q["video_id"].astype(str)
    p["video_id"] = p["video_id"].astype(str)

    if "split" in p.columns and prediction_split is not None:
        p = p[p["split"].astype(str) == str(prediction_split)].copy()

    if model_filter is not None and "model" in p.columns:
        p = p[p["model"].astype(str) == str(model_filter)].copy()

    required = {"video_id", "y_true", "y_prob", "y_pred"}
    missing = required - set(p.columns)
    if missing:
        raise KeyError(f"predictions_df missing required columns: {missing}")

    p["y_true"] = p["y_true"].astype(int)
    p["y_prob"] = p["y_prob"].astype(float)
    p["y_pred"] = p["y_pred"].astype(int)

    if aggregate_predictions_per_video:
        agg_cols = {
            "y_true": "first",
            "y_prob": "mean",
        }
        p = p.groupby("video_id", as_index=False).agg(agg_cols)
        p["y_pred"] = (p["y_prob"] >= 0.5).astype(int)

    p["abs_error_prob"] = np.abs(p["y_true"].astype(float) - p["y_prob"].astype(float))
    p["incorrect"] = (p["y_true"].astype(int) != p["y_pred"].astype(int)).astype(float)

    merge_cols = ["video_id", "y_true", "y_prob", "y_pred", "abs_error_prob", "incorrect"]
    for optional in ["dataset", "model", "seed", "repeat", "fold"]:
        if optional in p.columns:
            merge_cols.append(optional)

    merged = pd.merge(q, p[merge_cols], on="video_id", how="inner", suffixes=("_quality", "_pred"))

    if len(merged) == 0:
        return pd.DataFrame()

    if quality_cols is None:
        excluded = {
            "dataset",
            "dataset_quality",
            "dataset_pred",
            "video_id",
            "label",
            "use_active_features",
            "modality",
            "y_true",
            "y_prob",
            "y_pred",
            "abs_error_prob",
            "incorrect",
            "model",
            "seed",
            "repeat",
            "fold",
        }
        quality_cols = [
            c for c in merged.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(merged[c])
        ]

    rows = []
    targets = ["abs_error_prob", "incorrect"]

    group_cols = []
    if "dataset_quality" in merged.columns:
        group_cols.append("dataset_quality")
    elif "dataset" in merged.columns:
        group_cols.append("dataset")
    if "modality" in merged.columns:
        group_cols.append("modality")
    if "model" in merged.columns:
        group_cols.append("model")

    if group_cols:
        grouped_iter = merged.groupby(group_cols, dropna=False)
    else:
        grouped_iter = [((), merged)]

    for keys, g in grouped_iter:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)

        key_data = {}
        if group_cols:
            key_data = {col: val for col, val in zip(group_cols, keys)}

        for qc in quality_cols:
            x = pd.to_numeric(g[qc], errors="coerce")
            for target in targets:
                y = pd.to_numeric(g[target], errors="coerce")
                tmp = pd.DataFrame({"x": x, "y": y}).dropna()
                tmp = tmp[np.isfinite(tmp["x"]) & np.isfinite(tmp["y"])]

                if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
                    corr = float("nan")
                else:
                    corr = float(tmp["x"].corr(tmp["y"], method="pearson"))

                row = dict(key_data)
                row.update(
                    {
                        "quality_metric": qc,
                        "prediction_error_metric": target,
                        "n": int(len(tmp)),
                        "pearson_r": corr,
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# CORAL and cross-dataset generalization
# ============================================================

def _cov_sqrt_inv_sqrt(
    cov: np.ndarray,
    *,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return covariance square root and inverse square root via eigen decomposition."""
    cov = np.asarray(cov, dtype=np.float64)
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, float(eps))
    sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return sqrt, inv_sqrt


def coral_transform_full(
    X_source: np.ndarray,
    X_target_adapt: np.ndarray,
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Full CORAL transform source features toward target adaptation distribution.

    WARNING:
    Full CORAL uses dense covariance eigendecomposition and is not recommended
    for high-dimensional 13,527-D temporal-stat features.
    """
    Xs = np.asarray(X_source, dtype=np.float64)
    Xt = np.asarray(X_target_adapt, dtype=np.float64)

    if len(Xs) < 2 or len(Xt) < 2:
        warnings.warn("CORAL received fewer than 2 samples. Returning source unchanged.", RuntimeWarning)
        return Xs.astype(np.float32)

    mean_s = Xs.mean(axis=0, keepdims=True)
    mean_t = Xt.mean(axis=0, keepdims=True)

    Xs_c = Xs - mean_s
    Xt_c = Xt - mean_t

    cov_s = np.cov(Xs_c, rowvar=False) + float(eps) * np.eye(Xs.shape[1])
    cov_t = np.cov(Xt_c, rowvar=False) + float(eps) * np.eye(Xt.shape[1])

    sqrt_t, _ = _cov_sqrt_inv_sqrt(cov_t, eps=eps)
    _, inv_sqrt_s = _cov_sqrt_inv_sqrt(cov_s, eps=eps)

    Xs_coral = Xs_c @ inv_sqrt_s @ sqrt_t + mean_t
    return Xs_coral.astype(np.float32)


def coral_transform_diagonal(
    X_source: np.ndarray,
    X_target_adapt: np.ndarray,
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Diagonal CORAL / moment matching.

    Aligns source mean and per-feature standard deviation to target adaptation distribution.
    This is the default for high-dimensional temporal-stat features.
    """
    Xs = np.asarray(X_source, dtype=np.float64)
    Xt = np.asarray(X_target_adapt, dtype=np.float64)

    if len(Xs) < 1 or len(Xt) < 1:
        warnings.warn("Diagonal CORAL received empty input. Returning source unchanged.", RuntimeWarning)
        return Xs.astype(np.float32)

    mean_s = Xs.mean(axis=0, keepdims=True)
    mean_t = Xt.mean(axis=0, keepdims=True)

    std_s = Xs.std(axis=0, keepdims=True)
    std_t = Xt.std(axis=0, keepdims=True)

    Xs_aligned = (Xs - mean_s) / (std_s + float(eps)) * (std_t + float(eps)) + mean_t
    return Xs_aligned.astype(np.float32)


def coral_transform(
    X_source: np.ndarray,
    X_target_adapt: np.ndarray,
    *,
    eps: float = 1e-5,
    mode: str = "diagonal",
    full_dim_threshold: int = 2048,
) -> np.ndarray:
    """
    CORAL transform source features toward target adaptation distribution.

    IMPORTANT:
    X_target_adapt must be target train+val or another unlabeled adaptation set.
    It must not be target test.

    mode:
    - "diagonal": default, scalable mean/std alignment.
    - "full": full covariance CORAL. Automatically falls back to diagonal when feature dim is too large.
    """
    mode = str(mode).lower()
    Xs = np.asarray(X_source)
    n_features = int(Xs.shape[1]) if Xs.ndim == 2 else 0

    if mode in {"diagonal", "diag"}:
        return coral_transform_diagonal(X_source, X_target_adapt, eps=eps)

    if mode == "full":
        if n_features > int(full_dim_threshold):
            warnings.warn(
                f"Full CORAL requested with feature_dim={n_features}, exceeding threshold={full_dim_threshold}. "
                "Using diagonal CORAL instead to avoid memory blow-up.",
                RuntimeWarning,
            )
            return coral_transform_diagonal(X_source, X_target_adapt, eps=eps)
        return coral_transform_full(X_source, X_target_adapt, eps=eps)

    raise ValueError("mode must be 'diagonal' or 'full'.")


def _make_source_internal_split(
    video_ids: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Make source train/val split for cross-dataset source thresholding."""
    video_ids = np.asarray(video_ids).astype(str)
    y = np.asarray(y).astype(int)

    try:
        train_ids, val_ids, _, _ = train_test_split(
            video_ids,
            y,
            test_size=float(val_ratio),
            random_state=int(seed),
            stratify=y,
        )
    except ValueError:
        train_ids, val_ids, _, _ = train_test_split(
            video_ids,
            y,
            test_size=float(val_ratio),
            random_state=int(seed),
            stratify=None,
        )

    return {
        "train_videos": [str(v) for v in train_ids],
        "val_videos": [str(v) for v in val_ids],
    }


def run_cross_dataset_methods(
    *,
    source_dataset_name: str,
    target_dataset_name: str,
    source_data: Dict[str, Any],
    target_data: Dict[str, Any],
    target_splits: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    seed: int = 42,
    source_val_ratio: float = 0.2,
    models: Optional[Dict[str, Any]] = None,
    run_coral: bool = True,
    coral_mode: str = "diagonal",
    feature_mode: str = "temporal_stat",
) -> pd.DataFrame:
    """
    Run cross-dataset temporal-stat methods.

    Methods:
    - Temporal-stat source-only
    - Temporal-stat CORAL

    CORAL protocol:
    - Fit scaler on source train.
    - Use target train+val as unlabeled adaptation distribution.
    - Never use target test for adaptation.
    - Threshold is selected on source validation.
    - Evaluation is on target test.
    """
    if models is None:
        models = {
            "LogReg_temporal_stat": LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=int(seed),
            )
        }

    output_dir = Path(output_dir)
    ensure_dir(output_dir / "fold_results")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "tables")

    Xs, ys, vids_s = extract_video_features_for_ml(
        source_data,
        feature_mode=feature_mode,
        use_active_features=True,
    )
    Xt, yt, vids_t = extract_video_features_for_ml(
        target_data,
        feature_mode=feature_mode,
        use_active_features=True,
    )

    source_vid_to_idx = {str(v): i for i, v in enumerate(vids_s)}
    target_vid_to_idx = {str(v): i for i, v in enumerate(vids_t)}

    source_split = _make_source_internal_split(
        vids_s,
        ys,
        val_ratio=source_val_ratio,
        seed=seed,
    )
    source_train_idx = [source_vid_to_idx[str(v)] for v in source_split["train_videos"]]
    source_val_idx = [source_vid_to_idx[str(v)] for v in source_split["val_videos"]]

    rows = []

    for model_name, clf_template in models.items():
        for target_split in target_splits:
            repeat = int(target_split["repeat"])
            fold = int(target_split["fold"])
            fold_seed = int(seed) + (repeat - 1) * 100 + fold

            target_adapt_videos = list(target_split["train_videos"]) + list(target_split["val_videos"])
            target_test_videos = list(target_split["test_videos"])

            target_adapt_idx = [target_vid_to_idx[str(v)] for v in target_adapt_videos]
            target_test_idx = [target_vid_to_idx[str(v)] for v in target_test_videos]

            Xs_train, ys_train = Xs[source_train_idx], ys[source_train_idx]
            Xs_val, ys_val = Xs[source_val_idx], ys[source_val_idx]

            Xt_adapt = Xt[target_adapt_idx]
            Xt_test, yt_test = Xt[target_test_idx], yt[target_test_idx]

            scaler = StandardScaler()
            Xs_train_base = scaler.fit_transform(Xs_train)
            Xs_val_base = scaler.transform(Xs_val)
            Xt_adapt_base = scaler.transform(Xt_adapt)
            Xt_test_base = scaler.transform(Xt_test)

            for method_suffix, use_coral in [
                ("SourceOnly", False),
                ("CORAL", True),
            ]:
                if use_coral and not run_coral:
                    continue

                method_name = f"{model_name}_{method_suffix}"
                clf = _clone_and_set_random_state(clf_template, fold_seed)

                if use_coral:
                    Xs_train_fit = coral_transform(
                        Xs_train_base,
                        Xt_adapt_base,
                        mode=coral_mode,
                    )
                    Xs_val_eval = coral_transform(
                        Xs_val_base,
                        Xt_adapt_base,
                        mode=coral_mode,
                    )
                    coral_used = True
                else:
                    Xs_train_fit = Xs_train_base
                    Xs_val_eval = Xs_val_base
                    coral_used = False

                train_start = time.time()
                clf.fit(Xs_train_fit, ys_train)
                train_time_sec = time.time() - train_start

                source_val_prob = _safe_predict_proba_positive(clf, Xs_val_eval)
                threshold = choose_youden_threshold(ys_val, source_val_prob)

                target_prob = _safe_predict_proba_positive(clf, Xt_test_base)
                target_metrics = evaluate_at_threshold(yt_test, target_prob, threshold)
                inference_timing = _measure_sklearn_inference_time(clf, Xt_test_base)

                row = {
                    "dataset": target_dataset_name,
                    "source_dataset": source_dataset_name,
                    "target_dataset": target_dataset_name,
                    "model": method_name,
                    "seed": fold_seed,
                    "repeat": repeat,
                    "fold": fold,
                    "feature_mode": feature_mode,
                    "feature_dim": int(Xs.shape[1]),
                    "threshold": float(threshold),
                    "threshold_source": "source_validation_youden",
                    "train_time_sec": float(train_time_sec),
                    "n_params": np.nan,
                    "n_trainable_params": np.nan,
                    "n_total_params": np.nan,
                    "n_source_train_videos": int(len(source_train_idx)),
                    "n_source_val_videos": int(len(source_val_idx)),
                    "n_target_adapt_videos": int(len(target_adapt_idx)),
                    "n_target_test_videos": int(len(target_test_idx)),
                    "ms_per_video": float(inference_timing.get("ms_per_video", float("nan"))),
                    "coral_used": bool(coral_used),
                    "coral_mode": coral_mode if coral_used else "",
                    "coral_uses_target_test": False,
                }

                for k, v in target_metrics.items():
                    row[f"test_{k}"] = v

                rows.append(row)

                save_fold_results(
                    row,
                    output_dir / "fold_results" / f"{method_name}_{source_dataset_name}_to_{target_dataset_name}.csv",
                )

                pred_path = output_dir / "predictions" / f"{method_name}_{source_dataset_name}_to_{target_dataset_name}_video_predictions.csv"
                save_predictions(
                    pred_path,
                    dataset=target_dataset_name,
                    model_name=method_name,
                    seed=fold_seed,
                    repeat=repeat,
                    fold=fold,
                    split="test",
                    video_ids=vids_t[target_test_idx],
                    y_true=yt_test,
                    y_prob=target_prob,
                    threshold=threshold,
                    threshold_source="source_validation_youden",
                    source_dataset=source_dataset_name,
                    target_dataset=target_dataset_name,
                )

    out = pd.DataFrame(rows)
    out.to_csv(
        output_dir / "tables" / f"cross_dataset_{feature_mode}_{source_dataset_name}_to_{target_dataset_name}_summary.csv",
        index=False,
    )
    return out

def run_cross_dataset_mstgnet_source_only(
    *,
    source_dataset_name: str,
    target_dataset_name: str,
    source_data: Dict[str, Any],
    target_data: Dict[str, Any],
    target_splits: List[Dict[str, Any]],
    model_factory: Callable[[], nn.Module],
    device: torch.device,
    output_dir: Union[str, Path],
    seed: int = 42,
    source_val_ratio: float = 0.2,
    seq_len: int = 50,
    stride: int = 25,
    batch_size: int = 16,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    gradient_clip: float = 5.0,
    num_workers: int = 0,
    model_name: str = "MSTGNet_SourceOnly",
    save_checkpoint: bool = True,
) -> pd.DataFrame:
    """
    Run MSTGNet source-only cross-dataset transfer.

    Protocol:
    - Source dataset is split into source train/source validation.
    - Scaler is fit on source train sequences only.
    - Early stopping and threshold selection use source validation only.
    - Target train+val may define repeated target adaptation folds, but is NOT used for training.
    - Evaluation is on each target split's target test videos.
    - Target test is never used for scaling, training, thresholding, or adaptation.

    This supports the paper's cross-dataset source-only MSTGNet experiment.
    """
    set_all_seeds(seed)

    output_dir = Path(output_dir)
    ensure_dir(output_dir / "fold_results")
    ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "logs")

    Xs_seq, ys_seq, vids_s_seq = prepare_sequences(
        source_data,
        seq_len=seq_len,
        stride=stride,
        use_active_features=True,
        verbose=False,
    )
    Xt_seq, yt_seq, vids_t_seq = prepare_sequences(
        target_data,
        seq_len=seq_len,
        stride=stride,
        use_active_features=True,
        verbose=False,
    )

    source_video_ids, source_y_video = get_video_labels_from_data(source_data)

    source_split_simple = _make_source_internal_split(
        source_video_ids,
        source_y_video,
        val_ratio=source_val_ratio,
        seed=seed,
    )

    source_split = {
        "repeat": 1,
        "fold": 1,
        "seed": int(seed),
        "train_videos": source_split_simple["train_videos"],
        "val_videos": source_split_simple["val_videos"],
        # Dummy test not used for source training; keep nonempty if possible.
        "test_videos": source_split_simple["val_videos"],
    }

    source_data_split = split_sequence_data_by_video(
        Xs_seq,
        ys_seq,
        vids_s_seq,
        source_split,
    )
    Xs_train, ys_train, vids_s_train = source_data_split["train"]
    Xs_val, ys_val, vids_s_val = source_data_split["val"]

    if len(Xs_train) == 0 or len(Xs_val) == 0:
        raise ValueError(
            f"Source cross-dataset split is empty: train={len(Xs_train)}, val={len(Xs_val)}"
        )

    scaler = fit_train_scaler(Xs_train)
    Xs_train_s = apply_scaler(scaler, Xs_train)
    Xs_val_s = apply_scaler(scaler, Xs_val)

    train_loader = make_loader(
        Xs_train_s,
        ys_train,
        vids_s_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = make_loader(
        Xs_val_s,
        ys_val,
        vids_s_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )

    model = model_factory().to(device)
    n_params = count_parameters(model)
    n_total_params = count_total_parameters(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_val_auc = float("nan")
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    logs = []
    train_start = time.time()

    print("\n" + "=" * 70)
    print(f"Cross-dataset MSTGNet source-only: {source_dataset_name} -> {target_dataset_name}")
    print("=" * 70)
    print(
        f"Source sequences: train={len(Xs_train_s)}, val={len(Xs_val_s)} | "
        f"Source videos: train={len(source_split['train_videos'])}, val={len(source_split['val_videos'])}"
    )

    for epoch in range(1, int(epochs) + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_clip=gradient_clip,
        )

        val_vids, val_y_video, val_probs_video = predict_video_probs(
            model,
            val_loader,
            device,
        )
        val_auc = safe_auc(val_y_video, val_probs_video)

        logs.append(
            {
                "source_dataset": source_dataset_name,
                "target_dataset": target_dataset_name,
                "dataset": target_dataset_name,
                "model": model_name,
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": float(loss),
                "source_val_auc": float(val_auc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = False
        if best_state is None:
            improved = True
        elif np.isfinite(val_auc) and (
            (not np.isfinite(best_val_auc)) or val_auc > best_val_auc
        ):
            improved = True

        if improved:
            best_val_auc = float(val_auc) if np.isfinite(val_auc) else float("nan")
            best_epoch = int(epoch)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"  epoch={epoch:03d} | loss={_format_float(loss)} | "
                f"source_val_auc={_format_float(val_auc)} | "
                f"best={_format_float(best_val_auc)} | bad_epochs={bad_epochs}"
            )

        if bad_epochs >= int(patience):
            print(
                f"  Early stopping at epoch={epoch}. "
                f"Best epoch={best_epoch}, best source val AUC={_format_float(best_val_auc)}"
            )
            break

    train_time_sec = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        warnings.warn("best_state is None. Using last epoch model.", RuntimeWarning)

    source_val_vids, source_val_y_video, source_val_probs_video = predict_video_probs(
        model,
        val_loader,
        device,
    )
    threshold = choose_youden_threshold(source_val_y_video, source_val_probs_video)
    source_val_metrics = evaluate_at_threshold(
        source_val_y_video,
        source_val_probs_video,
        threshold,
    )

    ckpt_path = output_dir / "checkpoints" / f"{model_name}_{source_dataset_name}_to_{target_dataset_name}_seed{seed}.pt"
    if save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "threshold": float(threshold),
                "threshold_source": "source_validation_youden",
                "best_val_auc": float(best_val_auc),
                "best_epoch": int(best_epoch),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "source_dataset": source_dataset_name,
                "target_dataset": target_dataset_name,
                "dataset": target_dataset_name,
                "model_name": model_name,
                "seed": int(seed),
                "source_split": source_split,
                "n_params": int(n_params),
                "n_trainable_params": int(n_params),
                "n_total_params": int(n_total_params),
            },
            ckpt_path,
        )

    pd.DataFrame(logs).to_csv(
        output_dir / "logs" / f"{model_name}_{source_dataset_name}_to_{target_dataset_name}_seed{seed}_trainlog.csv",
        index=False,
    )

    rows = []

    for target_split in target_splits:
        repeat = int(target_split["repeat"])
        fold = int(target_split["fold"])
        fold_seed = int(seed) + (repeat - 1) * 100 + fold

        target_test_videos = set(map(str, target_split["test_videos"]))
        target_vids_all = np.asarray([str(v) for v in vids_t_seq])
        test_mask = np.array([v in target_test_videos for v in target_vids_all])

        Xt_test = Xt_seq[test_mask]
        yt_test = yt_seq[test_mask]
        vids_t_test = target_vids_all[test_mask]

        if len(Xt_test) == 0:
            warnings.warn(
                f"No target test sequences for repeat={repeat}, fold={fold}; skipped.",
                RuntimeWarning,
            )
            continue

        Xt_test_s = apply_scaler(scaler, Xt_test)

        test_loader = make_loader(
            Xt_test_s,
            yt_test,
            vids_t_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=fold_seed,
        )

        target_vids, target_y_video, target_probs_video = predict_video_probs(
            model,
            test_loader,
            device,
        )
        target_metrics = evaluate_at_threshold(
            target_y_video,
            target_probs_video,
            threshold,
        )

        inference_timing = measure_inference_time(
            model,
            test_loader,
            device,
            n_warmup_batches=1,
        )
        inference_timing_video = measure_inference_time_video(
            model,
            test_loader,
            device,
            n_warmup_batches=1,
        )

        row = {
            "dataset": target_dataset_name,
            "source_dataset": source_dataset_name,
            "target_dataset": target_dataset_name,
            "model": model_name,
            "seed": fold_seed,
            "repeat": repeat,
            "fold": fold,
            "threshold": float(threshold),
            "threshold_source": "source_validation_youden",
            "train_time_sec": float(train_time_sec),
            "n_params": int(n_params),
            "n_trainable_params": int(n_params),
            "n_total_params": int(n_total_params),
            "best_epoch": int(best_epoch),
            "best_source_val_auc": float(best_val_auc),
            "n_source_train_seq": int(len(Xs_train_s)),
            "n_source_val_seq": int(len(Xs_val_s)),
            "n_target_test_seq": int(len(Xt_test_s)),
            "n_source_train_videos": int(len(source_split["train_videos"])),
            "n_source_val_videos": int(len(source_split["val_videos"])),
            "n_target_test_videos": int(len(target_split["test_videos"])),
            "ms_per_batch": float(inference_timing.get("ms_per_batch", float("nan"))),
            "ms_per_sequence": float(inference_timing.get("ms_per_sequence", float("nan"))),
            "ms_per_video": float(inference_timing_video.get("ms_per_video", float("nan"))),
            "checkpoint_path": str(ckpt_path) if save_checkpoint else "",
            "uses_target_train_for_training": False,
            "uses_target_val_for_threshold": False,
            "uses_target_test_for_adaptation": False,
        }

        for k, v in source_val_metrics.items():
            row[f"source_val_{k}"] = v

        for k, v in target_metrics.items():
            row[f"test_{k}"] = v

        rows.append(row)

        save_fold_results(
            row,
            output_dir / "fold_results" / f"{model_name}_{source_dataset_name}_to_{target_dataset_name}.csv",
        )

        pred_path = output_dir / "predictions" / f"{model_name}_{source_dataset_name}_to_{target_dataset_name}_video_predictions.csv"
        save_predictions(
            pred_path,
            dataset=target_dataset_name,
            model_name=model_name,
            seed=fold_seed,
            repeat=repeat,
            fold=fold,
            split="test",
            video_ids=target_vids,
            y_true=target_y_video,
            y_prob=target_probs_video,
            threshold=threshold,
            threshold_source="source_validation_youden",
            source_dataset=source_dataset_name,
            target_dataset=target_dataset_name,
        )

    out = pd.DataFrame(rows)
    out.to_csv(
        output_dir / "tables" / f"cross_dataset_mstgnet_{source_dataset_name}_to_{target_dataset_name}_summary.csv",
        index=False,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out


# ============================================================
# Interpretability
# ============================================================

def compute_gradient_importance(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 16,
    feature_names: Optional[List[str]] = None,
    modality_slices: Optional[Dict[str, List[int]]] = None,
    max_sequences: Optional[int] = 2048,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute input-gradient feature importance using positive/deceptive logit.

    IMPORTANT:
    This uses gradient of the positive logit, not BCE loss.

    Parameters
    ----------
    max_sequences:
        If provided, randomly samples up to this many sequences to control runtime.
        Set None to use all sequences.
    """
    model = model.to(device)
    model.eval()

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X must have shape [N, T, F].")

    if max_sequences is not None and len(X) > int(max_sequences):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(X), size=int(max_sequences), replace=False)
        idx = np.sort(idx)
        X = X[idx]

    n_features = int(X.shape[-1])
    grad_sum = np.zeros(n_features, dtype=np.float64)
    n_batches = 0

    for start in range(0, len(X), int(batch_size)):
        xb_np = X[start:start + int(batch_size)]
        xb = torch.tensor(xb_np, dtype=torch.float32, device=device, requires_grad=True)

        model.zero_grad(set_to_none=True)
        logits = model(xb)

        objective = logits.sum()
        objective.backward()

        grad = xb.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy()
        grad_sum += grad.astype(np.float64)
        n_batches += 1

        del xb, logits, objective
        if device.type == "cuda":
            torch.cuda.empty_cache()

    importance = grad_sum / max(n_batches, 1)
    total = float(np.sum(importance))
    if total > 0:
        importance_norm = importance / total
    else:
        importance_norm = importance

    rows = []
    for i in range(n_features):
        row = {
            "feature_idx": int(i),
            "importance": float(importance[i]),
            "importance_norm": float(importance_norm[i]),
        }

        if feature_names is not None and i < len(feature_names):
            row["feature_name"] = str(feature_names[i])
        else:
            row["feature_name"] = f"feature_{i}"

        if modality_slices is not None:
            mod_name = ""
            for mod, sl in modality_slices.items():
                start_i, end_i = int(sl[0]), int(sl[1])
                if start_i <= i < end_i:
                    mod_name = mod
                    break
            row["modality"] = mod_name

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def save_gradient_importance(
    importance_df: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    dataset_name: str,
    model_name: str,
    top_k: int = 100,
    suffix: str = "",
) -> Dict[str, str]:
    """Save gradient importance to interpretability/."""
    output_dir = Path(output_dir)
    interp_dir = ensure_dir(output_dir / "interpretability")

    safe_model = str(model_name).replace("/", "_").replace(" ", "_")
    safe_dataset = str(dataset_name).replace("/", "_").replace(" ", "_")
    suffix = str(suffix).strip()
    suffix_part = f"_{suffix}" if suffix else ""

    full_path = interp_dir / f"gradient_importance_{safe_model}_{safe_dataset}{suffix_part}.csv"
    top_path = interp_dir / f"gradient_importance_top{int(top_k)}_{safe_model}_{safe_dataset}{suffix_part}.csv"
    modality_path = interp_dir / f"gradient_importance_modality_{safe_model}_{safe_dataset}{suffix_part}.csv"

    importance_df.to_csv(full_path, index=False)
    importance_df.head(int(top_k)).to_csv(top_path, index=False)

    paths = {
        "full": str(full_path),
        "top": str(top_path),
    }

    if "modality" in importance_df.columns:
        modality_df = (
            importance_df.groupby("modality", dropna=False)
            .agg(
                importance_sum=("importance", "sum"),
                importance_norm_sum=("importance_norm", "sum"),
                n_features=("feature_idx", "count"),
                mean_rank=("rank", "mean"),
            )
            .reset_index()
            .sort_values("importance_sum", ascending=False)
        )
        modality_df.to_csv(modality_path, index=False)
        paths["modality"] = str(modality_path)

    return paths


def compute_gradient_importance_from_checkpoint(
    *,
    checkpoint_path: Union[str, Path],
    model_factory: Callable[[], nn.Module],
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    device: torch.device,
    feature_names: Optional[List[str]] = None,
    modality_slices: Optional[Dict[str, List[int]]] = None,
    split_videos: Optional[Iterable[str]] = None,
    batch_size: int = 16,
    max_sequences: Optional[int] = 2048,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load checkpoint and compute gradient importance on selected sequences.

    If split_videos is provided, importance is computed only on sequences whose
    video_id is in split_videos. Otherwise, all X_seq is used.
    """
    model, scaler, _threshold, ckpt = load_model_checkpoint_for_inference(
        checkpoint_path=checkpoint_path,
        model_factory=model_factory,
        device=device,
    )

    X = np.asarray(X_seq, dtype=np.float32)
    vids = np.asarray(video_seq_ids).astype(str)

    if split_videos is not None:
        split_set = set(map(str, split_videos))
        mask = np.array([v in split_set for v in vids])
        X = X[mask]

    if len(X) == 0:
        raise ValueError(f"No sequences selected for checkpoint {checkpoint_path}")

    X = apply_scaler(scaler, X)

    imp = compute_gradient_importance(
        model,
        X,
        device=device,
        batch_size=batch_size,
        feature_names=feature_names,
        modality_slices=modality_slices,
        max_sequences=max_sequences,
        seed=seed,
    )

    imp["checkpoint_path"] = str(checkpoint_path)
    imp["dataset"] = ckpt.get("dataset", "")
    imp["model"] = ckpt.get("model_name", "")
    imp["seed"] = int(ckpt.get("seed", seed))
    imp["repeat"] = int(ckpt.get("repeat", -1))
    imp["fold"] = int(ckpt.get("fold", -1))
    imp["best_val_auc"] = float(ckpt.get("best_val_auc", np.nan))

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return imp


def compute_gradient_importance_across_checkpoints(
    *,
    checkpoint_paths: Iterable[Union[str, Path]],
    model_factory: Callable[[], nn.Module],
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    device: torch.device,
    feature_names: Optional[List[str]] = None,
    modality_slices: Optional[Dict[str, List[int]]] = None,
    use_checkpoint_val_split: bool = True,
    batch_size: int = 16,
    max_sequences: Optional[int] = 2048,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute gradient importance for multiple checkpoints.

    If use_checkpoint_val_split=True, each checkpoint uses its validation videos
    stored in ckpt["split"]["val_videos"]. This avoids using test labels for
    interpretability model selection.
    """
    all_dfs = []

    for i, ckpt_path in enumerate(checkpoint_paths):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            warnings.warn(f"Checkpoint not found and skipped: {ckpt_path}", RuntimeWarning)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        split_videos = None
        if use_checkpoint_val_split:
            split = ckpt.get("split", {})
            split_videos = split.get("val_videos", None)

        imp = compute_gradient_importance_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_factory=model_factory,
            X_seq=X_seq,
            y_seq=y_seq,
            video_seq_ids=video_seq_ids,
            device=device,
            feature_names=feature_names,
            modality_slices=modality_slices,
            split_videos=split_videos,
            batch_size=batch_size,
            max_sequences=max_sequences,
            seed=int(seed) + i,
        )
        all_dfs.append(imp)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def aggregate_gradient_importance_stability(
    importance_all_df: pd.DataFrame,
    *,
    top_k_values: Tuple[int, ...] = (10, 30, 100),
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate gradient importance stability across folds/checkpoints.

    Output includes:
    - mean_importance
    - std_importance
    - mean_importance_norm
    - mean_rank
    - top{k}_frequency
    - top{k}_rate
    - n_folds/checkpoints
    """
    if importance_all_df is None or len(importance_all_df) == 0:
        return pd.DataFrame()

    df = importance_all_df.copy()

    if group_cols is None:
        group_cols = ["feature_idx", "feature_name"]
        if "modality" in df.columns:
            group_cols.append("modality")

    required = {"feature_idx", "importance", "importance_norm", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"importance_all_df missing required columns: {missing}")

    instance_cols = [c for c in ["dataset", "model", "seed", "repeat", "fold", "checkpoint_path"] if c in df.columns]
    if instance_cols:
        n_instances = (
            df[instance_cols]
            .drop_duplicates()
            .shape[0]
        )
    else:
        n_instances = int(df.groupby("feature_idx").size().max())

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            mean_importance_norm=("importance_norm", "mean"),
            std_importance_norm=("importance_norm", "std"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            n_observed=("rank", "count"),
        )
        .reset_index()
    )

    for k in top_k_values:
        k = int(k)
        top_df = df[df["rank"] <= k]
        freq = (
            top_df.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name=f"top{k}_frequency")
        )
        agg = pd.merge(agg, freq, on=group_cols, how="left")
        agg[f"top{k}_frequency"] = agg[f"top{k}_frequency"].fillna(0).astype(int)
        agg[f"top{k}_rate"] = agg[f"top{k}_frequency"].astype(float) / max(int(n_instances), 1)

    agg["n_instances"] = int(n_instances)
    agg = agg.sort_values(
        ["mean_importance", "mean_importance_norm"],
        ascending=False,
    ).reset_index(drop=True)
    agg["stability_rank"] = np.arange(1, len(agg) + 1)

    return agg


def save_gradient_stability(
    stability_df: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    dataset_name: str,
    model_name: str,
    top_k: int = 100,
) -> Dict[str, str]:
    """Save gradient stability table."""
    output_dir = Path(output_dir)
    interp_dir = ensure_dir(output_dir / "interpretability")

    safe_model = str(model_name).replace("/", "_").replace(" ", "_")
    safe_dataset = str(dataset_name).replace("/", "_").replace(" ", "_")

    full_path = interp_dir / f"gradient_stability_{safe_model}_{safe_dataset}.csv"
    top_path = interp_dir / f"gradient_stability_top{int(top_k)}_{safe_model}_{safe_dataset}.csv"

    stability_df.to_csv(full_path, index=False)
    stability_df.head(int(top_k)).to_csv(top_path, index=False)

    return {
        "full": str(full_path),
        "top": str(top_path),
    }


# ============================================================
# Sanity checks and dry-run helpers
# ============================================================

def sanity_check_sequences_and_splits(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    splits: List[Dict[str, Any]],
    *,
    expected_input_dim: Optional[int] = 1503,
) -> pd.DataFrame:
    """Sanity-check sequence tensor, labels, and split overlap/nonempty status."""
    if X_seq.ndim != 3:
        raise ValueError(f"X_seq must be 3D [N,T,F], got shape {X_seq.shape}")

    if len(X_seq) != len(y_seq) or len(X_seq) != len(video_seq_ids):
        raise ValueError("X_seq, y_seq, and video_seq_ids length mismatch.")

    if expected_input_dim is not None and int(X_seq.shape[-1]) != int(expected_input_dim):
        warnings.warn(
            f"Input dim={X_seq.shape[-1]}, expected={expected_input_dim}. "
            "Continuing because dataset-driven metadata may be authoritative.",
            RuntimeWarning,
        )

    if not np.isfinite(X_seq).all():
        raise AssertionError("X_seq contains non-finite values.")

    if len(np.unique(y_seq)) < 2:
        warnings.warn("y_seq contains fewer than two classes.", RuntimeWarning)

    rows = []
    overlap_df = audit_video_split_overlap(splits)

    for s in splits:
        repeat = int(s["repeat"])
        fold = int(s["fold"])

        data_split = split_sequence_data_by_video(X_seq, y_seq, video_seq_ids, s)
        X_train, y_train, _ = data_split["train"]
        X_val, y_val, _ = data_split["val"]
        X_test, y_test, _ = data_split["test"]

        overlap_row = overlap_df[
            (overlap_df["repeat"] == repeat) & (overlap_df["fold"] == fold)
        ]

        status = "PASS"
        problems = []

        if len(X_train) == 0:
            status = "FAIL"
            problems.append("empty_train")
        if len(X_val) == 0:
            status = "FAIL"
            problems.append("empty_val")
        if len(X_test) == 0:
            status = "FAIL"
            problems.append("empty_test")

        if len(np.unique(y_train)) < 2:
            problems.append("single_class_train")
        if len(np.unique(y_val)) < 2:
            problems.append("single_class_val")
        if len(np.unique(y_test)) < 2:
            problems.append("single_class_test")

        if len(overlap_row) and str(overlap_row["status"].iloc[0]) != "PASS":
            status = "FAIL"
            problems.append("video_overlap")

        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "status": status,
                "problems": ";".join(problems),
                "n_train_seq": int(len(X_train)),
                "n_val_seq": int(len(X_val)),
                "n_test_seq": int(len(X_test)),
                "n_train_videos": int(len(s["train_videos"])),
                "n_val_videos": int(len(s["val_videos"])),
                "n_test_videos": int(len(s["test_videos"])),
                "train_class_0": int(np.sum(y_train == 0)),
                "train_class_1": int(np.sum(y_train == 1)),
                "val_class_0": int(np.sum(y_val == 0)),
                "val_class_1": int(np.sum(y_val == 1)),
                "test_class_0": int(np.sum(y_test == 0)),
                "test_class_1": int(np.sum(y_test == 1)),
            }
        )

    return pd.DataFrame(rows)


def dry_run_one_fold(
    *,
    dataset_name: str,
    model_name: str,
    model_factory: Callable[[], nn.Module],
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    video_seq_ids: np.ndarray,
    splits: List[Dict[str, Any]],
    device: torch.device,
    output_dir: Union[str, Path],
    batch_size: int = 8,
    epochs: int = 2,
    patience: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    gradient_clip: float = 5.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a short one-fold dry run to validate end-to-end training/evaluation pipeline.
    """
    if splits is None or len(splits) == 0:
        raise ValueError("splits must contain at least one split for dry run.")

    sanity_df = sanity_check_sequences_and_splits(
        X_seq,
        y_seq,
        video_seq_ids,
        splits[:1],
        expected_input_dim=int(X_seq.shape[-1]),
    )

    if len(sanity_df) == 0 or str(sanity_df["status"].iloc[0]) != "PASS":
        raise AssertionError(f"Dry-run sanity check failed:\n{sanity_df}")

    dry_output_dir = Path(output_dir) / "dry_run"
    ensure_dir(dry_output_dir)

    row = train_one_fold(
        dataset_name=dataset_name,
        model_name=f"{model_name}_DRYRUN",
        model_factory=model_factory,
        split=splits[0],
        X_seq=X_seq,
        y_seq=y_seq,
        video_seq_ids=video_seq_ids,
        device=device,
        output_dir=dry_output_dir,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        seed=seed,
        save_checkpoint=False,
    )

    auc_ok = np.isfinite(float(row.get("test_auc", np.nan)))
    row["dry_run_status"] = "PASS" if auc_ok else "PASS_WITH_NAN_AUC"
    row["dry_run_output_dir"] = str(dry_output_dir)
    return row


# ============================================================
# Convenience loaders/savers for result files
# ============================================================

def load_all_fold_results(
    output_dir: Union[str, Path],
    *,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load and concatenate all fold_results CSV files."""
    output_dir = Path(output_dir)
    fold_dir = output_dir / "fold_results"

    if not fold_dir.exists():
        return pd.DataFrame()

    files = sorted(fold_dir.glob(pattern))
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = str(f)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to read {f}: {e}", RuntimeWarning)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_all_predictions(
    output_dir: Union[str, Path],
    *,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load and concatenate all prediction CSV files."""
    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"

    if not pred_dir.exists():
        return pd.DataFrame()

    files = sorted(pred_dir.glob(pattern))
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = str(f)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to read {f}: {e}", RuntimeWarning)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def save_summary_tables(
    results_df: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    filename_prefix: str = "summary",
    metric_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Save fold-level summary and grouped summary tables."""
    output_dir = Path(output_dir)
    table_dir = ensure_dir(output_dir / "tables")

    fold_path = table_dir / f"{filename_prefix}_fold_results.csv"
    summary_path = table_dir / f"{filename_prefix}_grouped_summary.csv"

    results_df.to_csv(fold_path, index=False)

    summary_df = summarize_cv_results(
        results_df,
        metric_cols=metric_cols if metric_cols is not None else DEFAULT_METRIC_COLS,
        group_cols=["dataset", "model"],
    )
    summary_df.to_csv(summary_path, index=False)

    return {
        "fold_results": str(fold_path),
        "grouped_summary": str(summary_path),
    }


def save_repeat_level_ci_table(
    results_df: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    filename: str = "repeat_level_ci.csv",
    metrics: Optional[List[str]] = None,
    group_cols: Optional[List[str]] = None,
    ci: float = 0.95,
) -> str:
    """Save repeat-level CI table for one or more metrics."""
    if metrics is None:
        metrics = DEFAULT_METRIC_COLS
    if group_cols is None:
        group_cols = ["dataset", "model"]

    output_dir = Path(output_dir)
    table_dir = ensure_dir(output_dir / "tables")

    dfs = []
    for metric in metrics:
        if metric in results_df.columns:
            dfs.append(
                compute_repeat_level_ci(
                    results_df,
                    metric=metric,
                    group_cols=group_cols,
                    ci=ci,
                )
            )

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    path = table_dir / filename
    out.to_csv(path, index=False)
    return str(path)


def standardize_model_names_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add/overwrite paper-friendly model names.

    This does not change the raw 'model' column; it adds 'paper_model'.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    mapping = {
        "LogReg_temporal_stat": "Logistic Regression",
        "RF_temporal_stat": "Random Forest",
        "SVM_temporal_stat": "SVM (RBF)",
        "GradBoost_temporal_stat": "Gradient Boosting",
        "LogReg_mean": "Logistic Regression (mean)",
        "RF_mean": "Random Forest (mean)",
        "SVM_mean": "SVM (RBF, mean)",
        "GradBoost_mean": "Gradient Boosting (mean)",
        "CNNLSTM": "CNN-LSTM",
        "CNN-LSTM": "CNN-LSTM",
        "Transformer_L4": "Transformer",
        "MSTGNet": "MSTGNet",
    }

    out = df.copy()
    out["paper_model"] = out["model"].astype(str).map(mapping).fillna(out["model"].astype(str))
    return out


# ============================================================
# End of model_base.py
# ============================================================
