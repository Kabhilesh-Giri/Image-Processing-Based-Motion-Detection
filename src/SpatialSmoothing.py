from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import cv2


class ISpatialFilter(ABC):
    """Strategy interface for 2D spatial smoothing filters."""

    @abstractmethod
    def apply(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Apply spatial smoothing to a single grayscale frame.

        Parameters
        ----------
        gray_frame : np.ndarray
            2D grayscale image.

        Returns
        -------
        np.ndarray
            Smoothed 2D grayscale image (same shape).
        """
        raise NotImplementedError


class BoxFilter(ISpatialFilter):
    """
    Box (mean) filter with configurable kernel size (e.g., 3 or 5).
    """

    def __init__(self, kernel_size: int, border_type: int = cv2.BORDER_REFLECT_101):
        if kernel_size not in (3, 5):
            raise ValueError("BoxFilter kernel_size must be 3 or 5 for this project.")
        self._kernel_size = kernel_size
        self._border_type = border_type

    def apply(self, gray_frame: np.ndarray) -> np.ndarray:
        if gray_frame.ndim != 2:
            raise ValueError("BoxFilter expects a single-channel (2D) grayscale frame.")
        # cv2.blur performs mean filtering
        return cv2.blur(
            gray_frame,
            (self._kernel_size, self._kernel_size),
            borderType=self._border_type,
        )


class Gaussian2DFilter(ISpatialFilter):
    """
    2D Gaussian smoothing with configurable sigma and (optional) fixed kernel size.

    If kernel_size is None, it is derived from sigma using:
        k = 2 * ceil(3*sigma) + 1
    """

    def __init__(
        self,
        sigma: float,
        kernel_size: Optional[int] = None,
        border_type: int = cv2.BORDER_REFLECT_101,
    ):
        if sigma <= 0:
            raise ValueError("Gaussian2DFilter sigma must be > 0.")

        self._sigma = float(sigma)
        self._border_type = border_type

        if kernel_size is None:
            k = int(2 * np.ceil(3.0 * self._sigma) + 1)
            # Keep it at least 3 and odd
            k = max(3, k)
            if k % 2 == 0:
                k += 1
            self._kernel_size = k
        else:
            if kernel_size not in (3, 5):
                raise ValueError(
                    "For this project, Gaussian kernel_size must be 3 or 5 (or None for sigma-derived)."
                )
            self._kernel_size = kernel_size

    def apply(self, gray_frame: np.ndarray) -> np.ndarray:
        if gray_frame.ndim != 2:
            raise ValueError(
                "Gaussian2DFilter expects a single-channel (2D) grayscale frame."
            )
        return cv2.GaussianBlur(
            gray_frame,
            (self._kernel_size, self._kernel_size),
            sigmaX=self._sigma,
            sigmaY=self._sigma,
            borderType=self._border_type,
        )


class SpatialSmoothing:
    """
    Pipeline stage:
    - Accepts stacked grayscale frames
    - Applies the selected spatial smoothing strategy frame-by-frame
    - Returns stacked smoothed frames

    Input:
      frames: list[np.ndarray] or np.ndarray of shape (T, H, W)
    Output:
      smoothed_frames: np.ndarray of shape (T, H, W)
    """

    def __init__(self, spatial_filter: ISpatialFilter):
        if spatial_filter is None:
            raise ValueError(
                "SpatialSmoothing requires a valid ISpatialFilter strategy."
            )
        self._filter = spatial_filter

    def process(self, gray_frames: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        gray_frames : np.ndarray
            Stacked grayscale frames of shape (T, H, W).

        Returns
        -------
        np.ndarray
            Smoothed stacked frames of shape (T, H, W).
        """
        if not isinstance(gray_frames, np.ndarray):
            raise TypeError("gray_frames must be a numpy array of shape (T, H, W).")
        if gray_frames.ndim != 3:
            raise ValueError("gray_frames must have shape (T, H, W).")
        if gray_frames.shape[0] < 1:
            raise ValueError("gray_frames must contain at least 1 frame.")

        smoothed_list: List[np.ndarray] = []
        for t in range(gray_frames.shape[0]):
            frame = gray_frames[t]
            smoothed = self._filter.apply(frame)
            smoothed_list.append(smoothed)

        # Stack along time dimension
        return np.stack(smoothed_list, axis=0)
