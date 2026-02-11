from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional
import numpy as np
import cv2


class ThresholdStrategy(ABC):
    @abstractmethod
    def apply(self, deriv_frame: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        deriv_frame : np.ndarray
            2D derivative magnitude frame (expected uint8 in [0,255]).

        Returns
        -------
        np.ndarray
            2D thresholded binary mask (uint8, values 0 or 255).
        """
        raise NotImplementedError


class ManualThreshold(ThresholdStrategy):
    """
    Manual (fixed) threshold:
    mask = (deriv_frame >= T) ? 255 : 0
    """

    def __init__(self, threshold_value: int):
        if threshold_value < 0 or threshold_value > 255:
            raise ValueError("threshold_value must be in [0, 255].")
        self._T = int(threshold_value)

    def apply(self, deriv_frame: np.ndarray) -> np.ndarray:
        if not isinstance(deriv_frame, np.ndarray):
            raise TypeError("deriv_frame must be a numpy array.")
        if deriv_frame.ndim != 2:
            raise ValueError("deriv_frame must be a 2D grayscale image.")
        # If upstream ever changes dtype, we handle it without crashing.
        if deriv_frame.dtype != np.uint8:
            deriv_u8 = np.clip(deriv_frame, 0, 255).astype(np.uint8)
        else:
            deriv_u8 = deriv_frame

        mask = (deriv_u8 >= self._T).astype(np.uint8) * 255
        return mask


class AdaptiveThreshold(ThresholdStrategy):
    """
    OpenCV adaptive threshold.

    Notes:
    - Expects uint8 input.
    - block_size must be odd and >= 3.
    - C is a constant subtracted from the mean/gaussian mean.
    """

    def __init__(
        self,
        method: str = "gaussian",  # "mean" or "gaussian"
        block_size: int = 11,
        C: int = 2,
    ):
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("block_size must be odd and >= 3.")

        method_lower = method.lower().strip()
        if method_lower not in ("mean", "gaussian"):
            raise ValueError("method must be 'mean' or 'gaussian'.")

        self._block_size = int(block_size)
        self._C = int(C)
        self._method = method_lower

    def apply(self, deriv_frame: np.ndarray) -> np.ndarray:
        if not isinstance(deriv_frame, np.ndarray):
            raise TypeError("deriv_frame must be a numpy array.")
        if deriv_frame.ndim != 2:
            raise ValueError("deriv_frame must be a 2D grayscale image.")

        if deriv_frame.dtype != np.uint8:
            deriv_u8 = np.clip(deriv_frame, 0, 255).astype(np.uint8)
        else:
            deriv_u8 = deriv_frame

        adaptive_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self._method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )

        # Note: THRESH_BINARY expects bright pixels as foreground.
        mask = cv2.adaptiveThreshold(
            deriv_u8,
            maxValue=255,
            adaptiveMethod=adaptive_method,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self._block_size,
            C=self._C,
        )
        return mask


class Threshold:
    """
    Pipeline stage:
    - Consumes (idx, deriv_frame) iterator
    - Produces (idx, mask_frame) iterator

    This keeps the pipeline streaming and avoids storing everything in memory.
    """

    def __init__(self, strategy: ThresholdStrategy):
        if strategy is None:
            raise ValueError("strategy must not be None.")
        self._strategy = strategy

    def apply(
        self, deriv_iter: Iterator[Tuple[int, np.ndarray]]
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Parameters
        ----------
        deriv_iter : iterator yielding (idx, deriv_frame)

        Yields
        ------
        (idx, mask_frame) where mask_frame is uint8 {0,255}
        """
        for idx, deriv_frame in deriv_iter:
            mask = self._strategy.apply(deriv_frame)
            yield idx, mask
