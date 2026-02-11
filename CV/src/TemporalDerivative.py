import cv2
from abc import ABC, abstractmethod
import numpy as np
from pyparsing import deque
import scipy.ndimage


class TemporalDerivativeStrategy(ABC):
    @property
    @abstractmethod
    def window_size(self) -> int:
        pass

    @abstractmethod
    def compute(self, frames):
        """
        frames: list of grayscale frames, length == window_size
        returns: derivative frame
        """
        pass


class SimpleTemporalDifference(TemporalDerivativeStrategy):
    @property
    def window_size(self):
        return 2

    def compute(self, frames):
        if len(frames) != 2:
            raise ValueError("SimpleTemporalDifference requires exactly 2 frames")

        prev_frame = frames[0]
        curr_frame = frames[1]

        if prev_frame is None or curr_frame is None:
            raise ValueError("frames must not contain None")

        if prev_frame.shape != curr_frame.shape:
            raise ValueError("Frame shapes must match")

        if len(prev_frame.shape) != 2:
            raise ValueError("Expected grayscale frames (single-channel)")

        return cv2.absdiff(curr_frame, prev_frame)


class OneDDerivativeOfGaussian(TemporalDerivativeStrategy):
    def __init__(self, sigma, k=3):
        self.sigma = sigma
        self.k = k

    @property
    def window_size(self):
        # used by runner to know when output is valid
        radius = int(np.ceil(self.k * self.sigma))
        return 2 * radius + 1

    def compute(self, frames):
        """
        frames: list of grayscale frames, length == window_size
        """
        stack = np.stack(frames, axis=0).astype(np.float32)

        temporal_derivative = scipy.ndimage.gaussian_filter1d(
            stack, sigma=self.sigma, axis=0, order=1
        )

        # center frame (aligned output)
        center = len(frames) // 2
        diff = np.abs(temporal_derivative[center])

        return np.clip(diff, 0, 255).astype(np.uint8)


class OneDCenteredDifference(TemporalDerivativeStrategy):
    """
    1D centered temporal derivative:
    0.5 * [ -1, 0, 1 ]
    """

    @property
    def window_size(self):
        return 3

    def compute(self, frames):
        if len(frames) != 3:
            raise ValueError("OneDCenteredDifference requires exactly 3 frames")

        prev = frames[0].astype(np.float32)
        curr = frames[1]  # unused, but kept for semantic clarity
        next_ = frames[2].astype(np.float32)

        if prev.shape != next_.shape:
            raise ValueError("Frame shapes must match")

        if len(prev.shape) != 2:
            raise ValueError("Expected grayscale frames")

        # centered difference
        deriv = 0.5 * (next_ - prev)

        # for visualization only (signed → magnitude)
        deriv = np.abs(deriv)

        return np.clip(deriv, 0, 255).astype(np.uint8)


class Temporal_Derivative:
    def __init__(self, strategy: TemporalDerivativeStrategy):
        if strategy is None:
            raise ValueError("strategy must not be None")
        self._strategy = strategy

    @property
    def window_size(self):
        return self._strategy.window_size

    def apply(self, frame_iter):
        """
        frame_iter yields (idx, gray_frame)
        yields (aligned_idx, derivative_frame)
        """
        w = self._strategy.window_size
        buf = deque(maxlen=w)
        idx_buf = deque(maxlen=w)

        for idx, frame in frame_iter:
            buf.append(frame)
            idx_buf.append(idx)

            if len(buf) < w:
                continue

            out = self._strategy.compute(list(buf))

            # alignment policy:
            # - odd windows: center index
            # - even windows (w=2): latest index
            aligned_idx = idx_buf[w // 2] if (w % 2 == 1) else idx_buf[-1]
            yield aligned_idx, out
