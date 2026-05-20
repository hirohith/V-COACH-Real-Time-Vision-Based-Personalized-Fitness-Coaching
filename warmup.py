from typing import Dict, Optional


class WarmupRoutine:
    """
    Sequences through a list of timed stretches.
    Call tick(elapsed_seconds_since_start) each frame to get current state.
    """

    STRETCHES = [
        {"name": "Arm circles",      "duration": 20, "cue": "Rotate arms slowly — big circles"},
        {"name": "Hip hinges",       "duration": 20, "cue": "Hinge at hip, keep back flat"},
        {"name": "Bodyweight squat", "duration": 30, "cue": "Full depth, no weight — feel the stretch"},
        {"name": "Shoulder rolls",   "duration": 15, "cue": "Roll shoulders back then forward"},
    ]

    def __init__(self):
        self._total_duration = sum(s["duration"] for s in self.STRETCHES)

    @property
    def total_duration(self) -> int:
        return self._total_duration

    def tick(self, elapsed: float) -> Dict:
        """
        Returns current stretch info dict:
        {
          "name": str,
          "cue": str,
          "seconds_remaining": float,
          "stretch_index": int,
          "stretch_count": int,
          "overall_progress": float  (0.0 – 1.0)
        }
        Returns None if warm-up is complete.
        """
        if elapsed >= self._total_duration:
            return None

        cumulative = 0.0
        for i, stretch in enumerate(self.STRETCHES):
            cumulative += stretch["duration"]
            if elapsed < cumulative:
                seconds_remaining = cumulative - elapsed
                stretch_elapsed   = stretch["duration"] - seconds_remaining
                stretch_progress  = stretch_elapsed / stretch["duration"]
                return {
                    "name":             stretch["name"],
                    "cue":              stretch["cue"],
                    "seconds_remaining": round(seconds_remaining, 1),
                    "stretch_index":    i,
                    "stretch_count":    len(self.STRETCHES),
                    "overall_progress": min(elapsed / self._total_duration, 1.0),
                    "stretch_progress": min(stretch_progress, 1.0),
                }
        return None

    def is_complete(self, elapsed: float) -> bool:
        return elapsed >= self._total_duration
