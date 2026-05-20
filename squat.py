import time
from typing import Any, Dict, List, Optional
from state_machine import ExerciseStateMachine, RepState, RepRecord
from form_scoring import FormScore, score_depth, score_alignment
from personalization import UserProfile, get_angle_tolerance

SQUAT_START_KNEE_MIN = 155
SQUAT_DOWN_KNEE_MAX  = 120
SQUAT_IDEAL_MIN      = 65
SQUAT_IDEAL_MAX      = 100
SQUAT_BACK_IDEAL     = 165
SQUAT_BACK_TOLERANCE = 25


class SquatExercise(ExerciseStateMachine):
    def __init__(self, profile=None, target_reps=10):
        super().__init__(target_reps=target_reps)
        self._profile = profile or UserProfile()
        self._tol     = get_angle_tolerance(1.0, self._profile)
        self._depth_achieved = 999.0
        self._last_coach_ts  = 0.0

    def _determine_state(self, angles: Dict, pose_frame: Any) -> RepState:
        knee = angles.get("avg_knee", 180)
        if self._state == RepState.IDLE:
            if knee > SQUAT_START_KNEE_MIN:
                return RepState.START
        elif self._state == RepState.START:
            if knee < SQUAT_DOWN_KNEE_MAX * self._tol:
                self._depth_achieved = knee
                return RepState.DOWN
            self._coach_start(angles)
        elif self._state == RepState.DOWN:
            if knee < self._depth_achieved:
                self._depth_achieved = knee
            self._coach_down(angles)
            if knee > SQUAT_DOWN_KNEE_MAX * self._tol + 15:
                return RepState.START
        return self._state

    def _score_rep(self, angles: Dict, duration: float) -> RepRecord:
        depth     = score_depth(self._depth_achieved, SQUAT_IDEAL_MIN/self._tol, SQUAT_IDEAL_MAX*self._tol)
        alignment = score_alignment(angles, [
            {"angle": "torso",    "ideal": SQUAT_BACK_IDEAL, "tolerance": SQUAT_BACK_TOLERANCE*self._tol, "weight": 0.7},
            {"angle": "avg_knee", "ideal": 90,               "tolerance": 25,                             "weight": 0.3},
        ])
        stability = self._stability_score()
        tips = []
        if depth < 25:     tips.append("Squat deeper — reach below parallel")
        if alignment < 18: tips.append("Keep your torso more upright")
        if stability < 18: tips.append("Slow down — control the movement")
        if not tips:       tips.append("Excellent squat!")
        self._depth_achieved = 999.0
        total = min(depth + alignment + stability, 100.0)
        return RepRecord(0, total, depth, alignment, stability, duration, coaching_notes=tips)

    def get_bad_joints(self, angles: Dict) -> List[str]:
        bad = []
        if angles.get("torso", 180) < SQUAT_BACK_IDEAL - SQUAT_BACK_TOLERANCE:
            bad += ["left_shoulder", "right_shoulder"]
        if angles.get("left_knee",  180) < 50: bad.append("left_knee")
        if angles.get("right_knee", 180) < 50: bad.append("right_knee")
        return bad

    def get_form_score_live(self, angles: Dict) -> FormScore:
        d = score_depth(angles.get("avg_knee", 180), SQUAT_IDEAL_MIN, SQUAT_IDEAL_MAX)
        a = score_alignment(angles, [{"angle": "torso", "ideal": SQUAT_BACK_IDEAL,
                                      "tolerance": SQUAT_BACK_TOLERANCE, "weight": 1.0}])
        return FormScore.from_parts(d, a, self._stability_score())

    def _coach_start(self, angles):
        if time.time() - self._last_coach_ts < 2.5: return
        self._last_coach_ts = time.time()
        self.push_coaching_message("When ready, squat down")

    def _coach_down(self, angles):
        if time.time() - self._last_coach_ts < 2.5: return
        self._last_coach_ts = time.time()
        if angles.get("torso", 180) < SQUAT_BACK_IDEAL - SQUAT_BACK_TOLERANCE:
            self.push_coaching_message("Keep your back straight!")
        elif angles.get("avg_knee", 180) > 105:
            self.push_coaching_message("Go lower — below parallel")
        else:
            self.push_coaching_message("Good depth — drive through heels")
