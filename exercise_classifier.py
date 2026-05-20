from collections import deque
from typing import Dict, Tuple


class ExerciseClassifier:
    """
    Rule-based exercise classifier with 30-frame majority vote for stability.
    predict() returns (exercise_name, confidence 0–1).
    """

    EXERCISES = ["Squat", "Push-up", "Shoulder Press", "Unknown"]

    def __init__(self, vote_window: int = 30, confidence_threshold: float = 0.75):
        self._vote_window  = vote_window
        self._conf_thresh  = confidence_threshold
        self._vote_buffer: deque = deque(maxlen=vote_window)

    def _rule_classify(self, angles: Dict) -> Tuple[str, float]:
        """
        Deterministic rules based on joint angles.
        Returns (label, confidence).
        """
        avg_elbow = angles.get("avg_elbow", 180)
        avg_knee  = angles.get("avg_knee",  180)
        torso     = angles.get("torso",     180)

        # Push-up: arms partially bent, body horizontal (torso angle low)
        if avg_elbow > 150 and torso < 130:
            return "Push-up", 0.90

        # Shoulder press: elbows bent below 120°, body upright
        if avg_elbow < 120 and torso > 150:
            return "Shoulder Press", 0.88

        # Squat: knee angle drops below 130°
        if avg_knee < 130:
            return "Squat", 0.92

        # Standing / ambiguous
        if avg_knee > 155 and torso > 155:
            # Arms extended overhead more likely press
            if avg_elbow > 155:
                return "Shoulder Press", 0.60
            return "Squat", 0.55

        return "Unknown", 0.40

    def predict(self, angles: Dict) -> Tuple[str, float]:
        """
        Classify current frame, add to vote buffer, return majority prediction.
        """
        if not angles:
            return "Unknown", 0.0

        label, raw_conf = self._rule_classify(angles)
        self._vote_buffer.append(label)

        if len(self._vote_buffer) < self._vote_window // 2:
            return label, raw_conf * 0.5

        # Majority vote
        counts = {ex: self._vote_buffer.count(ex) for ex in self.EXERCISES}
        winner = max(counts, key=counts.get)
        confidence = counts[winner] / len(self._vote_buffer)
        return winner, round(confidence, 2)

    def reset(self):
        self._vote_buffer.clear()

    def is_confident(self, angles: Dict) -> Tuple[bool, str, float]:
        """
        Returns (ready, exercise_name, confidence).
        ready=True when buffer is full and confidence > threshold.
        """
        name, conf = self.predict(angles)
        ready = (
            len(self._vote_buffer) >= self._vote_window
            and conf >= self._conf_thresh
            and name != "Unknown"
        )
        return ready, name, conf
