from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _safe_name(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "unknown"


def _save_frame_stack(folder: Path, frames, ext: str = ".png"):
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"frame_{i:05d}{ext}"), frame)


def _save_aligned_frames(folder: Path, aligned_indices, frames, ext: str = ".png"):
    folder.mkdir(parents=True, exist_ok=True)
    for j, (idx, frame) in enumerate(zip(aligned_indices, frames)):
        cv2.imwrite(str(folder / f"frameidx_{int(idx):05d}_seq_{j:05d}{ext}"), frame)


def build_aligned_streams(
    orig_frames,
    idx_list,
    deriv_list,
    mask_by_idx,
):
    aligned_indices = []
    aligned_deriv = []
    aligned_mask = []
    aligned_overlay = []

    for j, i in enumerate(idx_list):
        if i not in mask_by_idx:
            continue

        mask = mask_by_idx[i].astype(np.uint8)
        masked = cv2.bitwise_and(orig_frames[i], orig_frames[i], mask=mask)

        aligned_indices.append(i)
        aligned_deriv.append(deriv_list[j])
        aligned_mask.append(mask)
        aligned_overlay.append(masked)

    return aligned_indices, aligned_deriv, aligned_mask, aligned_overlay


def save_run_outputs(
    cfg,
    smoothing_name: str,
    td_name: str,
    thr_name: str,
    orig_frames,
    gray_frames,
    temporal_input_frames,
    aligned_indices,
    aligned_deriv,
    aligned_mask,
    aligned_overlay,
):
    output_root = Path(
        cfg.get("output_root")
        or (Path(cfg["database_root"]).parent / "Kabhilesh-Aidan-Project-Output")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    video_name = _safe_name(Path(cfg["video_path"]).name)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_folder_name = _safe_name(
        f"{run_tag}__{video_name}__s_{smoothing_name}__td_{td_name}__th_{thr_name}"
    )
    run_dir = output_root / run_folder_name

    orig_dir = run_dir / "original_frames"
    gray_dir = run_dir / "grayscale_frames"
    smooth_dir = run_dir / f"{_safe_name(smoothing_name)}_smoothing_frames"
    td_dir = run_dir / f"{_safe_name(td_name)}_temporal_derivatives_frames"
    thr_dir = run_dir / f"{_safe_name(thr_name)}_threshold_frames"
    overlay_dir = run_dir / "overlayed_frames"

    _save_frame_stack(orig_dir, orig_frames)
    _save_frame_stack(gray_dir, gray_frames)
    _save_frame_stack(smooth_dir, temporal_input_frames)
    _save_aligned_frames(td_dir, aligned_indices, aligned_deriv)
    _save_aligned_frames(thr_dir, aligned_indices, aligned_mask)
    _save_aligned_frames(overlay_dir, aligned_indices, aligned_overlay)

    return run_dir


def six_view_generator(
    orig_frames,
    gray_frames,
    temporal_input_frames,
    idx_list,
    deriv_list,
    mask_by_idx,
    deriv_to_u8,
):
    for j, i in enumerate(idx_list):
        if i not in mask_by_idx:
            continue

        orig = orig_frames[i]
        gray = gray_frames[i]
        smooth = temporal_input_frames[i]
        deriv = deriv_to_u8(deriv_list[j])
        mask = mask_by_idx[i].astype(np.uint8)
        masked = cv2.bitwise_and(orig, orig, mask=mask)

        yield i, [orig, gray, smooth, deriv, mask, masked], [
            "Original",
            "Grayscale",
            "Smoothing",
            "1D Derivative",
            "Threshold Mask",
            "Masked Output",
        ]
