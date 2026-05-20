import time
from typing import Any, Dict, List
from state_machine import ExerciseStateMachine, RepState, RepRecord
from form_scoring import FormScore, score_depth, score_alignment
from personalization import UserProfile, get_angle_tolerance

PU_START_ELBOW_MIN = 150
PU_DOWN_ELBOW_MAX  = 110
PU_IDEAL_MIN       = 70
PU_IDEAL_MAX       = 95
PU_TORSO_IDEAL     = 175
PU_TORSO_TOL       = 15


class PushupExercise(ExerciseStateMachine):
    def __init__(self, profile=None, target_reps=10):
        super().__init__(target_reps=target_reps)
        self._profile = profile or UserProfile()
        self._tol     = get_angle_tolerance(1.0, self._profile)
        self._depth_achieved = 999.0
        self._last_coach_ts  = 0.0

    def _determine_state(self, angles: Dict, pose_frame: Any) -> RepState:
        elbow = angles.get("avg_elbow", 180)
        if self._state == RepState.IDLE:
            if elbow > PU_START_ELBOW_MIN: return RepState.START
        elif self._state == RepState.START:
            if elbow < PU_DOWN_ELBOW_MAX * self._tol:
                self._depth_achieved = elbow; return RepState.DOWN
            self._coach_start()
        elif self._state == RepState.DOWN:
            if elbow < self._depth_achieved: self._depth_achieved = elbow
            self._coach_down(angles)
            if elbow > PU_DOWN_ELBOW_MAX * self._tol + 15: return RepState.START
        return self._state

    def _score_rep(self, angles: Dict, duration: float) -> RepRecord:
        depth     = score_depth(self._depth_achieved, PU_IDEAL_MIN/self._tol, PU_IDEAL_MAX*self._tol)
        alignment = score_alignment(angles, [
            {"angle": "torso", "ideal": PU_TORSO_IDEAL, "tolerance": PU_TORSO_TOL*self._tol, "weight": 0.6},
            {"angle": "spine", "ideal": 175,            "tolerance": 15,                      "weight": 0.4},
        ])
        stability = self._stability_score()
        tips = []
        if depth < 22:     tips.append("Lower your chest closer to the floor")
        if alignment < 18: tips.append("Maintain a straight plank throughout")
        if not tips:       tips.append("Great push-up! Full range of motion")
        self._depth_achieved = 999.0
        total = min(depth + alignment + stability, 100.0)
        return RepRecord(0, total, depth, alignment, stability, duration, coaching_notes=tips)

    def get_bad_joints(self, angles: Dict) -> List[str]:
        bad = []
        if angles.get("torso", 180) < PU_TORSO_IDEAL - PU_TORSO_TOL:
            bad += ["left_hip", "right_hip"]
        return bad

    def get_form_score_live(self, angles: Dict) -> FormScore:
        d = score_depth(angles.get("avg_elbow", 180), PU_IDEAL_MIN, PU_IDEAL_MAX)
        a = score_alignment(angles, [{"angle": "torso", "ideal": PU_TORSO_IDEAL,
                                      "tolerance": PU_TORSO_TOL, "weight": 1.0}])
        return FormScore.from_parts(d, a, self._stability_score())

    def _coach_start(self):
        if time.time() - self._last_coach_ts < 2.5: return
        self._last_coach_ts = time.time()
        self.push_coaching_message("Get into plank — arms extended")

    def _coach_down(self, angles):
        if time.time() - self._last_coach_ts < 2.5: return
        self._last_coach_ts = time.time()
        if angles.get("torso", 180) < PU_TORSO_IDEAL - PU_TORSO_TOL * 1.5:
            self.push_coaching_message("Keep hips level — rigid plank!")
        elif angles.get("avg_elbow", 180) > 100:
            self.push_coaching_message("Go lower — chest to floor")
        else:
            self.push_coaching_message("Good — push back up!")
