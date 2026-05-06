# gui.py
from __future__ import annotations

from typing import Callable, Dict, Any, Optional, Tuple
import ipywidgets as W
from IPython.display import clear_output
import threading
from pathlib import Path

import src.Utilities as utilities_mod


DEFAULT_DATABASE_ROOT = str(Path(__file__).resolve().parents[1] / "Database")


def build_gui(
    video_options,
    default_database_root: str = DEFAULT_DATABASE_ROOT,
    default_video: Optional[str] = None,
    default_smoothing: str = "box3",
    default_smoothing_sigma: float = 1.0,
    default_temporal: str = "op1d",
    default_temporal_sigma: float = 0.4,
    default_threshold: str = "manual",
    default_threshold_value: int = 25,
    on_run: Optional[Callable[[Dict[str, Any], W.Output], None]] = None,
) -> Tuple[
    W.VBox, Callable[[], Dict[str, Any]], Callable[[], Dict[str, Any]], W.Output
]:
    """
    Returns:
      ui: ipywidgets container to display
      get_config: function -> cfg dict (validates)
      wait_for_run: blocking function -> cfg dict (validates) after Run click
      out: output widget (prints errors/status)
    """

    # -----------------------------
    # Widgets
    # -----------------------------
    def _default_output_root(database_root: str) -> str:
        db = Path(database_root)
        parent = db.parent if db.parent != db else db
        return str(parent / "Kabhilesh-Aidan-Project-Output")

    db_root_text = W.Text(
        value=str(default_database_root),
        description="Database:",
        layout=W.Layout(width="800px"),
        style={"description_width": "initial"},
    )
    refresh_btn = W.Button(description="Refresh Videos", button_style="info")

    output_root_text = W.Text(
        value=_default_output_root(default_database_root),
        description="Output Root:",
        layout=W.Layout(width="800px"),
        style={"description_width": "initial"},
    )

    video_dd = W.Dropdown(
        options=video_options,
        value=(
            default_video
            if default_video is not None
            else (video_options[0] if video_options else None)
        ),
        description="Video Path:",
        layout=W.Layout(width="800px"),
    )

    smoothing_toggle = W.ToggleButtons(
        options=[
            ("3x3 Box", "box3"),
            ("5x5 Box", "box5"),
            ("Sigma", "sigma"),
            ("None", "none"),
        ],
        value=default_smoothing,
        description="Smoothing:",
        style={"description_width": "initial"},
    )

    smoothing_sigma = W.FloatText(
        value=float(default_smoothing_sigma),
        description="Smoothing Sigma:",
        style={"description_width": "initial"},
        layout=W.Layout(width="250px"),
    )
    smoothing_sigma_row = W.HBox([smoothing_sigma])
    smoothing_sigma_row.layout.display = "none"

    temporal_toggle = W.ToggleButtons(
        options=[
            ("1-D differential operator", "op1d"),
            ("simple 0.5[-1, 0, 1]", "centered"),
            ("1D derivative of a Gaussian", "dog"),
        ],
        value=default_temporal,
        description="Temporal:",
        style={"description_width": "initial"},
    )

    dog_sigma = W.FloatText(
        value=float(default_temporal_sigma),
        description="Temporal Sigma:",
        style={"description_width": "initial"},
        layout=W.Layout(width="250px"),
    )
    dog_row = W.HBox([dog_sigma])
    dog_row.layout.display = "none"

    # -----------------------------
    # NEW: Threshold section
    # -----------------------------
    threshold_toggle = W.ToggleButtons(
        options=[
            ("Manual", "manual"),
            ("Adaptive", "adaptive"),
        ],
        value=default_threshold,
        description="Threshold:",
        style={"description_width": "initial"},
    )

    manual_thr = W.IntText(
        value=int(default_threshold_value),
        description="Threshold Value:",
        style={"description_width": "initial"},
        layout=W.Layout(width="250px"),
    )
    manual_thr_row = W.HBox([manual_thr])
    manual_thr_row.layout.display = "none"

    run_btn = W.Button(description="Run", button_style="success")
    out = W.Output()

    def _refresh_videos_and_output():
        db_root = db_root_text.value.strip()
        output_root_text.value = _default_output_root(db_root)
        options = utilities_mod.list_sequence_folders(db_root)
        video_dd.options = options
        if options:
            video_dd.value = options[0]
        else:
            video_dd.value = None

    # -----------------------------
    # Show/hide logic
    # -----------------------------
    def _refresh_visibility() -> None:
        smoothing_sigma_row.layout.display = (
            "flex" if smoothing_toggle.value == "sigma" else "none"
        )
        dog_row.layout.display = "flex" if temporal_toggle.value == "dog" else "none"
        manual_thr_row.layout.display = (
            "flex" if threshold_toggle.value == "manual" else "none"
        )

    def on_smoothing_change(change):
        if change.get("name") == "value":
            _refresh_visibility()

    def on_temporal_change(change):
        if change.get("name") == "value":
            _refresh_visibility()

    def on_threshold_change(change):
        if change.get("name") == "value":
            _refresh_visibility()

    def on_refresh_clicked(_):
        out.clear_output(wait=True)
        try:
            _refresh_videos_and_output()
            with out:
                print(f"Loaded {len(video_dd.options)} sequence folders.")
        except Exception as e:
            with out:
                print("Failed to load database path:")
                print(e)

    smoothing_toggle.observe(on_smoothing_change, names="value")
    temporal_toggle.observe(on_temporal_change, names="value")
    threshold_toggle.observe(on_threshold_change, names="value")
    refresh_btn.on_click(on_refresh_clicked)
    _refresh_visibility()

    # -----------------------------
    # Validation
    # -----------------------------
    def get_config() -> Dict[str, Any]:
        if video_dd.value is None:
            raise ValueError("No video selected.")

        cfg: Dict[str, Any] = {
            "database_root": db_root_text.value.strip(),
            "video_path": video_dd.value,
            "output_root": output_root_text.value.strip(),
            "smoothing": {"mode": smoothing_toggle.value},
            "temporal": {"mode": temporal_toggle.value},
            "threshold": {"mode": threshold_toggle.value},
        }

        if not cfg["database_root"]:
            raise ValueError("Database path is required.")
        if not cfg["output_root"]:
            raise ValueError("Output root is required.")

        if smoothing_toggle.value == "sigma":
            s = float(smoothing_sigma.value)
            if s <= 0:
                raise ValueError("Smoothing sigma must be > 0")
            cfg["smoothing"]["smoothing_sigma"] = s

        if temporal_toggle.value == "dog":
            ts = float(dog_sigma.value)
            if ts <= 0:
                raise ValueError("Temporal sigma must be > 0")
            cfg["temporal"]["temporal_sigma"] = ts

        if threshold_toggle.value == "manual":
            t = int(manual_thr.value)
            if t < 0:
                raise ValueError("Manual threshold must be >= 0")
            cfg["threshold"]["threshold_value"] = t
        # if adaptive: do nothing

        return cfg

    # -----------------------------
    # "Return config when Run clicked"
    # -----------------------------
    done_evt = threading.Event()
    result_holder: Dict[str, Any] = {"cfg": None}

    def on_run_clicked(_):
        out.clear_output(wait=True)
        try:
            cfg = get_config()
            result_holder["cfg"] = cfg
            with out:
                for k, v in cfg.items():
                    print(f"{k}: {v}")
            if on_run is not None:
                on_run(cfg, out)
            done_evt.set()
        except Exception as e:
            with out:
                print("Invalid selection:")
                print(e)

    run_btn.on_click(on_run_clicked)

    def wait_for_run() -> Dict[str, Any]:
        """
        Blocks until user clicks Run with a valid config, then returns that config.
        Use in notebook only (Jupyter). Not for scripts.
        """
        done_evt.wait()
        return result_holder["cfg"]

    # -----------------------------
    # Layout
    # -----------------------------
    ui = W.VBox(
        [
            W.HBox([db_root_text, refresh_btn]),
            output_root_text,
            video_dd,
            W.HBox([smoothing_toggle, smoothing_sigma_row]),
            W.HBox([temporal_toggle, dog_row]),
            W.HBox([threshold_toggle, manual_thr_row]),
            run_btn,
            out,
        ]
    )

    return ui, get_config, wait_for_run, out
