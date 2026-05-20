import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class PoseFrame:
    landmarks: Optional[object]
    world_landmarks: Optional[object]
    landmark_dict: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    world_landmark_dict: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    visibility: Dict[str, float] = field(default_factory=dict)
    raw_image: Optional[np.ndarray] = None
    confidence: float = 0.0
    width: int = 640
    height: int = 480


class PoseDetector:
    LANDMARK_NAMES = {
        0:  "nose",
        11: "left_shoulder",  12: "right_shoulder",
        13: "left_elbow",     14: "right_elbow",
        15: "left_wrist",     16: "right_wrist",
        23: "left_hip",       24: "right_hip",
        25: "left_knee",      26: "right_knee",
        27: "left_ankle",     28: "right_ankle",
    }

    def __init__(self, model_complexity: int = 1, smoothing_window: int = 5):
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smoothing_window = smoothing_window
        self._landmark_history: List[Dict] = []

    def process_frame(self, frame: np.ndarray) -> PoseFrame:
        h, w = frame.shape[:2]
        enhanced = self._enhance(frame)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        pf = PoseFrame(
            landmarks=results.pose_landmarks,
            world_landmarks=results.pose_world_landmarks,
            raw_image=frame.copy(),
            width=w, height=h,
        )
        if results.pose_landmarks:
            pf.confidence = float(np.mean([lm.visibility for lm in results.pose_landmarks.landmark]))
            lm_dict, vis_dict = self._extract(results.pose_landmarks, w, h)
            pf.landmark_dict = self._smooth(lm_dict)
            pf.visibility    = vis_dict

        if results.pose_world_landmarks:
            pf.world_landmark_dict = self._extract_world(results.pose_world_landmarks)

        return pf

    def process_frame_dual(self, frame: np.ndarray) -> Tuple["PoseFrame", "PoseFrame"]:
        """
        Split frame at midpoint, detect pose on each half independently.
        Right-half x-coordinates are offset by +0.5 to map back to full-frame space.
        Returns (pf_left, pf_right).
        """
        h, w = frame.shape[:2]
        mid  = w // 2

        left_half  = frame[:, :mid]
        right_half = frame[:, mid:]

        pf_left  = self.process_frame(left_half)
        pf_left.width  = w
        pf_left.height = h

        # Process right half with a fresh Pose instance to avoid state conflicts
        pf_right_raw = self._process_half(right_half)

        # Remap right-half x-coords to full frame: x_full = (x_half * 0.5) + 0.5
        remapped: Dict[str, Tuple[float, float, float]] = {}
        for name, (x, y, z) in pf_right_raw.landmark_dict.items():
            remapped[name] = (x * 0.5 + 0.5, y, z)

        pf_right_raw.landmark_dict = remapped
        pf_right_raw.width  = w
        pf_right_raw.height = h
        pf_right_raw.raw_image = frame.copy()

        return pf_left, pf_right_raw

    def _process_half(self, half_frame: np.ndarray) -> "PoseFrame":
        """Stateless process on a sub-frame (uses a temporary Pose instance)."""
        h, w = half_frame.shape[:2]
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as tmp_pose:
            rgb = cv2.cvtColor(self._enhance(half_frame), cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = tmp_pose.process(rgb)

        pf = PoseFrame(
            landmarks=results.pose_landmarks,
            world_landmarks=results.pose_world_landmarks,
            raw_image=half_frame.copy(),
            width=w, height=h,
        )
        if results.pose_landmarks:
            pf.confidence = float(np.mean([lm.visibility for lm in results.pose_landmarks.landmark]))
            lm_dict, vis_dict = self._extract(results.pose_landmarks, w, h)
            pf.landmark_dict = lm_dict
            pf.visibility    = vis_dict
        return pf

    def draw_landmarks(self, pf: "PoseFrame", highlight_joints=None,
                       dot_color=(0, 200, 255)) -> np.ndarray:
        canvas = pf.raw_image.copy()
        if pf.landmarks is None:
            return canvas
        self.mp_drawing.draw_landmarks(
            canvas, pf.landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=dot_color, thickness=2, circle_radius=3),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(180, 180, 180), thickness=2),
        )
        for name in (highlight_joints or []):
            if name in pf.landmark_dict:
                x, y, _ = pf.landmark_dict[name]
                cx, cy  = int(x * pf.width), int(y * pf.height)
                cv2.circle(canvas, (cx, cy), 10, (0, 60, 255), -1)
                cv2.circle(canvas, (cx, cy), 12, (255, 255, 255), 2)
        return canvas

    def draw_landmarks_on(self, canvas: np.ndarray, pf: "PoseFrame",
                          dot_color=(0, 200, 255),
                          highlight_joints=None) -> np.ndarray:
        """Draw landmarks onto an existing canvas (for dual-person overlay)."""
        if pf.landmarks is None:
            return canvas
        h, w = canvas.shape[:2]
        for idx, name in self.LANDMARK_NAMES.items():
            if name not in pf.landmark_dict:
                continue
            x, y, _ = pf.landmark_dict[name]
            px, py  = int(x * w), int(y * h)
            color   = (0, 60, 255) if (highlight_joints and name in highlight_joints) else dot_color
            cv2.circle(canvas, (px, py), 5, color, -1)
            cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1)
        return canvas

    def release(self):
        self.pose.close()

    @staticmethod
    def _enhance(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def _extract(self, landmarks, w: int, h: int):
        lm_dict: Dict[str, Tuple[float, float, float]] = {}
        vis_dict: Dict[str, float] = {}
        for idx, name in self.LANDMARK_NAMES.items():
            lk = landmarks.landmark[idx]
            lm_dict[name]  = (lk.x, lk.y, lk.z)
            vis_dict[name] = lk.visibility
        return lm_dict, vis_dict

    def _extract_world(self, world_landmarks) -> Dict[str, Tuple[float, float, float]]:
        """Extract 3D metric world landmark coords."""
        wlm: Dict[str, Tuple[float, float, float]] = {}
        for idx, name in self.LANDMARK_NAMES.items():
            lk = world_landmarks.landmark[idx]
            wlm[name] = (lk.x, lk.y, lk.z)
        return wlm

    def _smooth(self, current: Dict) -> Dict:
        self._landmark_history.append(current)
        if len(self._landmark_history) > self._smoothing_window:
            self._landmark_history.pop(0)
        if len(self._landmark_history) == 1:
            return current
        weights = np.exp(np.linspace(-1, 0, len(self._landmark_history)))
        weights /= weights.sum()
        smoothed: Dict = {}
        for name in current:
            xs = [h[name][0] for h in self._landmark_history if name in h]
            ys = [h[name][1] for h in self._landmark_history if name in h]
            zs = [h[name][2] for h in self._landmark_history if name in h]
            n  = len(xs); w = weights[-n:]; w = w / w.sum()
            smoothed[name] = (float(np.dot(w, xs)), float(np.dot(w, ys)), float(np.dot(w, zs)))
        return smoothed
