import time
from typing import Any, Dict, List
from state_machine import ExerciseStateMachine, RepState, RepRecord
from form_scoring import FormScore, score_depth, score_alignment
from personalization import UserProfile, get_angle_tolerance

SP_START_MAX   = 100
SP_UP_MIN      = 155
SP_IDEAL_MIN   = 155
SP_IDEAL_MAX   = 175
SP_TORSO_IDEAL = 170
SP_TORSO_TOL   = 20


class ShoulderPressExercise(ExerciseStateMachine):
    def __init__(self, profile=None, target_reps=12):
        super().__init__(target_reps=target_reps)
        self._profile = profile or UserProfile()
        self._tol     = get_angle_tolerance(1.0, self._profile)
        self._peak_elbow     = 0.0
        self._last_coach_ts  = 0.0

    def _determine_state(self, angles: Dict, pose_frame: Any) -> RepState:
        elbow = angles.get("avg_elbow", 0)
        if self._state == RepState.IDLE:
            if elbow < SP_START_MAX * self._tol: return RepState.START
        elif self._state == RepState.START:
            if elbow > SP_UP_MIN / self._tol:
                self._peak_elbow = elbow; return RepState.DOWN
            self._coach_start()
        elif self._state == RepState.DOWN:
            if elbow > self._peak_elbow: self._peak_elbow = elbow
            self._coach_up(angles)
            if elbow < SP_START_MAX * self._tol + 20: return RepState.START
        return self._state

    def _score_rep(self, angles: Dict, duration: float) -> RepRecord:
        depth     = score_depth(self._peak_elbow, SP_IDEAL_MIN/self._tol, SP_IDEAL_MAX)
        alignment = score_alignment(angles, [
            {"angle": "torso",          "ideal": SP_TORSO_IDEAL, "tolerance": SP_TORSO_TOL*self._tol, "weight": 0.5},
            {"angle": "left_shoulder",  "ideal": 160,            "tolerance": 20,                     "weight": 0.25},
            {"angle": "right_shoulder", "ideal": 160,            "tolerance": 20,                     "weight": 0.25},
        ])
        stability = self._stability_score()
        tips = []
        if depth < 25:     tips.append("Press fully overhead — lock out arms")
        if alignment < 18: tips.append("Keep torso vertical — brace your core")
        if not tips:       tips.append("Perfect shoulder press!")
        self._peak_elbow = 0.0
        total = min(depth + alignment + stability, 100.0)
        return RepRecord(0, total, depth, alignment, stability, duration, coaching_notes=tips)

    def get_bad_joints(self, angles: Dict) -> List[str]:
        bad = []
        if angles.get("torso", 180) < SP_TORSO_IDEAL - SP_TORSO_TOL:
            bad += ["left_hip", "right_hip"]
        return bad

    def get_form_score_live(self, angles: Dict) -> FormScore:
        d = score_depth(angles.get("avg_elbow", 0), SP_IDEAL_MIN, SP_IDEAL_MAX)
        a = score_alignment(angles, [{"angle": "torso", "ideal": SP_TORSO_IDEAL,
                                      "tolerance": SP_TORSO_TOL, "weight": 1.0}])
        return FormScore.from_parts(d, a, self._stability_score())

    def _coach_start(self):
        if time.time() - self._last_coach_ts < 2.5: return
        self._last_coach_ts = time.time()
        self.push_coaching_message("Elbows at shoulder height — press up!")

    def _coach_up(self, angles):
        if time.time() - self._last_coach_ts < 2.5: return
        self._last_coach_ts = time.time()
        if angles.get("torso", 180) < SP_TORSO_IDEAL - SP_TORSO_TOL:
            self.push_coaching_message("Don't lean back — brace your core")
        elif angles.get("avg_elbow", 0) < 145:
            self.push_coaching_message("Fully extend — lock out overhead")
        else:
            self.push_coaching_message("Great! Lower under control")
