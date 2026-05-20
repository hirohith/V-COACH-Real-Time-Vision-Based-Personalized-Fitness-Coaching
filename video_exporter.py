import cv2
import os
import numpy as np
from collections import deque
from typing import Optional


class RepVideoExporter:
    """
    Records a short annotated clip around each completed rep.
    Maintains a rolling pre-buffer so the clip starts before the rep begins.
    """

    def __init__(self, output_dir: str, fps: int = 20,
                 buffer_before: int = 30, buffer_after: int = 15):
        self._output_dir   = output_dir
        self._fps          = fps
        self._buf_before   = buffer_before
        self._buf_after    = buffer_after

        os.makedirs(output_dir, exist_ok=True)

        self._pre_buffer: deque = deque(maxlen=buffer_before)
        self._writer: Optional[cv2.VideoWriter] = None
        self._post_frames_remaining = 0
        self._current_rep    = 0
        self._current_score  = 0.0
        self._current_path   = ""

    def add_frame(self, frame: np.ndarray):
        """Call every loop iteration with the annotated BGR frame."""
        self._pre_buffer.append(frame.copy())

        if self._writer is not None and self._post_frames_remaining > 0:
            self._writer.write(frame)
            self._post_frames_remaining -= 1
            if self._post_frames_remaining == 0:
                self._finish_clip()

    def on_rep_complete(self, rep_number: int, form_score: float):
        """
        Trigger a clip save for the completed rep.
        Flushes pre-buffer then records post_frames_after more frames.
        """
        # Close any in-progress clip first
        if self._writer is not None:
            self._finish_clip()

        self._current_rep   = rep_number
        self._current_score = form_score

        fname = f"rep_{rep_number:02d}_{int(form_score):03d}.mp4"
        self._current_path  = os.path.join(self._output_dir, fname)

        # Determine frame size from pre-buffer
        if not self._pre_buffer:
            return
        h, w = self._pre_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._current_path, fourcc, self._fps, (w, h))

        # Write pre-buffered frames
        for f in self._pre_buffer:
            self._writer.write(f)

        self._post_frames_remaining = self._buf_after

    def _finish_clip(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._post_frames_remaining = 0

    def close(self):
        self._finish_clip()

    @staticmethod
    def list_clips(output_dir: str):
        """Return sorted list of .mp4 clip paths."""
        if not os.path.exists(output_dir):
            return []
        return sorted(
            [os.path.join(output_dir, f)
             for f in os.listdir(output_dir) if f.endswith(".mp4")],
            reverse=True,
        )
