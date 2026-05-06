import re
import cv2
from pathlib import Path
import numpy as np


DEFAULT_DATABASE_ROOT = str(Path(__file__).resolve().parents[1] / "Database")


class ImageSequenceReader:
    def __init__(self, folder_path: str, extensions=(".jpg", ".jpeg", ".png")):
        self.folder = Path(folder_path)
        if not self.folder.exists() or not self.folder.is_dir():
            raise FileNotFoundError(f"Sequence folder not found: {folder_path}")

        self.extensions = tuple(e.lower() for e in extensions)
        self.files = self._collect_files()

        if not self.files:
            raise FileNotFoundError(f"No frames found in: {folder_path}")

    def _collect_files(self):
        files = []
        for e in self.extensions:
            files.extend(self.folder.glob(f"*{e}"))

        # numeric sort using the last integer in filename (works for ...0002.jpg etc.)
        def key(p: Path):
            m = re.findall(r"\d+", p.stem)
            return int(m[-1]) if m else 10**18

        return sorted(files, key=key)

    def frames_gray(self):
        frames = []

        for p in self.files:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            frames.append(gray)

        if len(frames) == 0:
            raise RuntimeError("No valid grayscale frames loaded.")

        return np.stack(frames, axis=0)

    def get_metadata(self) -> dict:
        # for image sequences, metadata is different
        first = cv2.imread(str(self.files[0]), cv2.IMREAD_GRAYSCALE)
        h, w = first.shape
        return {"frame_count": len(self.files), "width": w, "height": h}

    def frames_bgr_and_gray(self):
        """
        Loads BGR frames and derives grayscale from them so alignment is guaranteed.
        Returns:
        bgr_stack: (T, H, W, 3) uint8
        gray_stack: (T, H, W) uint8
        """
        bgr_frames = []
        gray_frames = []

        for p in self.files:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            bgr_frames.append(bgr)
            gray_frames.append(gray)

        if len(bgr_frames) == 0:
            raise RuntimeError("No valid frames loaded.")

        return np.stack(bgr_frames, axis=0), np.stack(gray_frames, axis=0)


def pick_sequence_folder(db_root=DEFAULT_DATABASE_ROOT) -> str:
    root = Path(db_root)
    if not root.exists():
        raise FileNotFoundError(f"Database folder not found: {db_root}")

    candidates = []

    for d in root.rglob("*"):
        if not d.is_dir():
            continue

        # HARD FILTER: ignore macOS junk folders
        if "__MACOSX" in d.parts:
            continue

        # must contain at least one jpg/jpeg/png
        if any(d.glob("*.jpg")) or any(d.glob("*.jpeg")) or any(d.glob("*.png")):
            candidates.append(d)

    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No valid image-sequence folders found under: {db_root}"
        )

    for i, d in enumerate(candidates):
        print(f"[{i}] {d}")

    choice = int(input("Select sequence folder index: "))
    if choice < 0 or choice >= len(candidates):
        raise ValueError("Invalid index.")

    return str(candidates[choice])
