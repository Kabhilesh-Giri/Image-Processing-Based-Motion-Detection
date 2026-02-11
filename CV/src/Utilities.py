import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import cv2
import numpy as np
import time
import math
import ipywidgets as widgets
from dataclasses import dataclass
from pathlib import Path


def play_side_by_side(frame_iter, fps=30, figsize=(12, 4), out=None):
    """
    frame_iter yields:
        (idx, frames, titles)

    frames : list of images (BGR, gray, or mask)
    titles : list of strings (same length as frames)
    """

    delay = 1.0 / fps

    fig = None
    axes = None
    ims = None
    try:
        for idx, frames, titles in frame_iter:
            if len(frames) != len(titles):
                raise ValueError("frames and titles length mismatch")

            n = len(frames)

            # Create figure ONCE
            if fig is None:
                fig, axes = plt.subplots(1, n, figsize=figsize)
                if n == 1:
                    axes = [axes]

                ims = []
                for ax in axes:
                    ax.axis("off")
                    im = ax.imshow(np.zeros((10, 10)), cmap="gray")
                    ims.append(im)

            for i, (frame, title) in enumerate(zip(frames, titles)):
                img = frame

                # BGR → RGB if needed
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ims[i].set_cmap(None)
                else:
                    ims[i].set_cmap("gray")
                    ims[i].set_clim(0, 255)

                ims[i].set_data(img)
                axes[i].set_title(title)

            if out is None:
                clear_output(wait=True)
                display(fig)
            else:
                out.clear_output(wait=True)
                with out:
                    display(fig)
            time.sleep(delay)

    except KeyboardInterrupt:
        # Clean stop without dumping traceback
        print("Stopped playback.")
    finally:
        if fig is not None:
            plt.close(fig)


def deriv_to_u8(d):
    d = np.asarray(d)
    if d.dtype == np.uint8:
        return d
    d = d.astype(np.float32)
    mn, mx = float(d.min()), float(d.max())
    if mx - mn < 1e-6:
        return np.zeros_like(d, dtype=np.uint8)
    d = (d - mn) * (255.0 / (mx - mn))
    return np.clip(d, 0, 255).astype(np.uint8)


def ensure_u8_mask(mask):
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() == 1:
        m = m * 255
    return m


def _to_u8_display(img):
    """
    Convert derivative/float/grayscale to uint8 0..255 for display.
    Safe for:
      - float derivative frames (negative/positive)
      - uint8 gray
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)

    # Map to 0..255 using min/max (per-frame). Good enough for visualization.
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < 1e-6:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - mn) * (255.0 / (mx - mn))
    return np.clip(a, 0, 255).astype(np.uint8)


def play_grid(
    frame_iter, fps=20, ncols=4, figsize_per_cell=(4.5, 3.0), max_frames=None
):
    delay = 1.0 / fps

    fig = None
    axes = None
    ims = None
    shown = 0

    try:
        for idx, frames, titles in frame_iter:
            if len(frames) != len(titles):
                raise ValueError("frames and titles length mismatch")

            k = len(frames)
            ncols_ = min(ncols, k)
            nrows_ = int(math.ceil(k / ncols_))

            if fig is None:
                fig_w = figsize_per_cell[0] * ncols_
                fig_h = figsize_per_cell[1] * nrows_
                fig, axes = plt.subplots(nrows_, ncols_, figsize=(fig_w, fig_h))
                axes = np.array(axes).reshape(-1)

                ims = []
                for ax in axes:
                    ax.axis("off")
                    im = ax.imshow(
                        np.zeros((10, 10), dtype=np.uint8),
                        cmap="gray",
                        vmin=0,
                        vmax=255,
                    )
                    ims.append(im)

            # Update panels
            for j in range(len(axes)):
                ax = axes[j]
                im = ims[j]

                if j < k:
                    im.set_data(frames[j])
                    ax.set_title(titles[j], fontsize=10)
                else:
                    im.set_data(np.zeros((10, 10), dtype=np.uint8))
                    ax.set_title("")

            clear_output(wait=True)
            display(fig)
            time.sleep(delay)

            shown += 1
            if max_frames is not None and shown >= max_frames:
                break

    except KeyboardInterrupt:
        print("Stopped playback.")
    finally:
        if fig is not None:
            plt.close(fig)


def multi_view_generator(
    orig_frames,
    gray_frames,
    idx_simple,
    deriv_simple,
    mask_simple,
    masked_simple,
    sigma_values,
    deriv_dog,
    mask_dog,
    masked_dog,
    deriv_to_u8,
):
    """
    Yields (idx, frames, titles) for grid playback.

    Note:
    - orig_frames: list of BGR frames (uint8)
    - gray_frames: list of gray frames (uint8)
    - idx_simple: list of indices for which derivative exists
    - deriv_simple: list of derivative frames aligned with idx_simple
    - mask_simple, masked_simple: lists aligned with idx_simple
    - sigma_values: list of sigmas
    - deriv_dog/mask_dog/masked_dog: dict sigma -> list aligned by position i
    - deriv_to_u8: function that converts derivative frame to uint8 for display
    """

    # shortest common length across streams
    n = len(idx_simple)
    for sigma in sigma_values:
        n = min(n, len(deriv_dog[sigma]), len(mask_dog[sigma]), len(masked_dog[sigma]))

    for i in range(n):
        idx = idx_simple[i]

        # Convert BGR -> RGB for matplotlib
        orig_rgb = cv2.cvtColor(orig_frames[idx], cv2.COLOR_BGR2RGB)
        gray = gray_frames[idx]

        simple_deriv_u8 = deriv_to_u8(deriv_simple[i])
        dog_deriv_u8 = [deriv_to_u8(deriv_dog[s][i]) for s in sigma_values]

        simple_m = mask_simple[i]
        dog_m = [mask_dog[s][i] for s in sigma_values]

        simple_masked_rgb = cv2.cvtColor(masked_simple[i], cv2.COLOR_BGR2RGB)
        dog_masked_rgb = [
            cv2.cvtColor(masked_dog[s][i], cv2.COLOR_BGR2RGB) for s in sigma_values
        ]

        frames = [
            orig_rgb,
            gray,
            simple_deriv_u8,
            *dog_deriv_u8,
            simple_m,
            *dog_m,
            simple_masked_rgb,
            *dog_masked_rgb,
        ]

        titles = [
            "Original",
            "Grayscale",
            "Simple Derivative 0.5[-1,0,1]",
            *(f"DoG Derivative σ={s}" for s in sigma_values),
            "Threshold (Simple)",
            *(f"Threshold (DoG σ={s})" for s in sigma_values),
            "Masked (Simple)",
            *(f"Masked (DoG σ={s})" for s in sigma_values),
        ]

        yield idx, frames, titles


@dataclass
class RunConfig:
    seq_folder: str | None = None
    sigma: float | None = None


# Module-level singleton config (lives inside Utilities.py)
_CFG = RunConfig()


def build_run_config_ui(root=r"F:\ML\CV\Database"):
    """
    Shows a simple UI:
      - dropdown for sequence folder
      - exact sigma input
      - Apply button

    Stores values in _CFG.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Database root not found: {root}")

    # Only top-level sequence folders, ignoring __MACOSX
    seq_folders = sorted(
        str(p) for p in root_path.iterdir() if p.is_dir() and "__MACOSX" not in p.name
    )

    if not seq_folders:
        raise RuntimeError("No valid sequence folders found")

    folder_dd = widgets.Dropdown(
        options=seq_folders,
        description="Sequence:",
        layout=widgets.Layout(width="800px"),
    )

    sigma_text = widgets.FloatText(
        value=1.0,
        description="Sigma:",
        layout=widgets.Layout(width="300px"),
    )

    apply_btn = widgets.Button(description="Apply", button_style="success")
    out = widgets.Output()

    def _apply(_):
        with out:
            out.clear_output()

            folder = folder_dd.value
            sigma = float(sigma_text.value)

            if sigma <= 0:
                raise ValueError("Sigma must be > 0")

            _CFG.seq_folder = folder
            _CFG.sigma = sigma

            print("Config set:")
            print(f"  folder: {_CFG.seq_folder}")
            print(f"  sigma : {_CFG.sigma}")

    apply_btn.on_click(_apply)

    display(widgets.VBox([folder_dd, sigma_text, apply_btn, out]))


def get_run_config() -> RunConfig:
    """
    Returns the config AFTER Apply.
    """
    if _CFG.seq_folder is None or _CFG.sigma is None:
        raise RuntimeError(
            "Config not set. Select values and click Apply, then re-run this cell."
        )
    return _CFG


def list_sequence_folders(db_root=r"F:\ML\CV\Database"):
    root = Path(db_root)
    if not root.exists():
        raise FileNotFoundError(f"Database folder not found: {db_root}")

    candidates = []

    for d in root.rglob("*"):
        if not d.is_dir():
            continue

        # ignore macOS junk
        if "__MACOSX" in d.parts:
            continue

        # must contain image files
        if any(d.glob("*.jpg")) or any(d.glob("*.jpeg")) or any(d.glob("*.png")):
            candidates.append(str(d))

    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No valid image-sequence folders found under: {db_root}"
        )

    return candidates
