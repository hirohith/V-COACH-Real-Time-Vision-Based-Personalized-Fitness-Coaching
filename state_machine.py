from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import numpy as np


class RepState(Enum):
    IDLE  = "IDLE"
    START = "START"
    DOWN  = "DOWN"
    UP    = "UP"


@dataclass
class RepRecord:
    rep_number: int
    form_score: float
    depth_score: float
    alignment_score: float
    stability_score: float
    duration_seconds: float
    eccentric_s: float = 0.0
    concentric_s: float = 0.0
    coaching_notes: List[str] = field(default_factory=list)


class ExerciseStateMachine:
    def __init__(self, target_reps: int = 10, stability_window: int = 8):
        self.target_reps = target_reps
        self._state = RepState.IDLE
        self._reps = 0
        self._rep_history: List[RepRecord] = []
        self._state_start = time.time()
        self._rep_start   = time.time()
        self._down_start_ts: float = 0.0
        self._down_end_ts: float   = 0.0
        self._stability_window = stability_window
        self._angle_history: List[Dict] = []
        self._coaching_queue: List[str] = []

    @property
    def state(self) -> RepState:
        return self._state

    @property
    def reps(self) -> int:
        return self._reps

    @property
    def rep_history(self) -> List[RepRecord]:
        return self._rep_history

    @property
    def latest_rep(self) -> Optional[RepRecord]:
        return self._rep_history[-1] if self._rep_history else None

    def update(self, angles: Dict, pose_frame: Any, confidence: float = 1.0) -> Optional[RepRecord]:
        self._angle_history.append(angles)
        if len(self._angle_history) > self._stability_window:
            self._angle_history.pop(0)
        if confidence < 0.45:
            self._set_state(RepState.IDLE)
            return None

        new_state = self._determine_state(angles, pose_frame)
        completed = None

        if new_state != self._state:
            if self._state == RepState.START and new_state == RepState.DOWN:
                self._rep_start   = time.time()
                self._down_start_ts = time.time()

            if self._state == RepState.DOWN and new_state in (RepState.UP, RepState.START):
                self._down_end_ts = time.time()
                completed = self._finish_rep(angles)

            self._set_state(new_state)

        return completed

    def reset(self):
        self._state = RepState.IDLE
        self._reps  = 0
        self._rep_history.clear()
        self._angle_history.clear()
        self._coaching_queue.clear()

    def pop_coaching_message(self) -> Optional[str]:
        return self._coaching_queue.pop(0) if self._coaching_queue else None

    def push_coaching_message(self, msg: str):
        if not self._coaching_queue or self._coaching_queue[-1] != msg:
            self._coaching_queue.append(msg)

    def check_fatigue(self) -> Optional[str]:
        """
        Inspect last 3 reps for fatigue signals.
        Returns a warning string or None.
        """
        reps = self._rep_history
        if len(reps) >= 3:
            scores = [r.form_score for r in reps[-3:]]
            if scores[0] - scores[-1] > 15:
                return "Form degrading — consider stopping"

        if len(reps) >= 2:
            last_two = reps[-2:]
            if all(r.eccentric_s < 1.0 and r.eccentric_s > 0 for r in last_two):
                return "Rushing — slow down"

        return None

    def _determine_state(self, angles: Dict, pose_frame: Any) -> RepState:
        raise NotImplementedError

    def _score_rep(self, angles: Dict, duration: float) -> RepRecord:
        raise NotImplementedError

    def _set_state(self, s: RepState):
        self._state = s
        self._state_start = time.time()

    def _finish_rep(self, angles: Dict) -> RepRecord:
        now      = time.time()
        duration = now - self._rep_start
        eccentric_s  = max(0.0, self._down_end_ts - self._down_start_ts)
        concentric_s = max(0.0, duration - eccentric_s)

        self._reps += 1
        rec = self._score_rep(angles, duration)
        rec.rep_number    = self._reps
        rec.duration_seconds = duration
        rec.eccentric_s   = round(eccentric_s,  2)
        rec.concentric_s  = round(concentric_s, 2)
        self._rep_history.append(rec)
        return rec

    def _stability_score(self) -> float:
        if len(self._angle_history) < 3:
            return 25.0
        variances = [
            float(np.var([h[k] for h in self._angle_history if k in h]))
            for k in self._angle_history[0]
        ]
        if not variances:
            return 25.0
        return round(max(0.0, 30.0 - (float(np.mean(variances)) / 200.0) * 30.0), 1)
