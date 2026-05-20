import numpy as np
from typing import Dict, Tuple, Optional


def angle_between(a, b, c, use_3d: bool = False) -> float:
    if use_3d:
        va = np.array(a) - np.array(b)
        vc = np.array(c) - np.array(b)
    else:
        va = np.array(a[:2]) - np.array(b[:2])
        vc = np.array(c[:2]) - np.array(b[:2])
    na, nc = np.linalg.norm(va), np.linalg.norm(vc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0))))


def mid(a, b) -> Tuple:
    return ((a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2)


def calculate_joint_angles(lm: Dict) -> Dict[str, float]:
    """2D angle calculation (normalised image coords)."""
    g = lambda n: lm.get(n, (0., 0., 0.))
    angles: Dict[str, float] = {}
    angles["left_knee"]      = angle_between(g("left_hip"),       g("left_knee"),      g("left_ankle"))
    angles["right_knee"]     = angle_between(g("right_hip"),      g("right_knee"),     g("right_ankle"))
    angles["left_hip"]       = angle_between(g("left_shoulder"),  g("left_hip"),       g("left_knee"))
    angles["right_hip"]      = angle_between(g("right_shoulder"), g("right_hip"),      g("right_knee"))
    angles["left_elbow"]     = angle_between(g("left_shoulder"),  g("left_elbow"),     g("left_wrist"))
    angles["right_elbow"]    = angle_between(g("right_shoulder"), g("right_elbow"),    g("right_wrist"))
    angles["left_shoulder"]  = angle_between(g("left_elbow"),     g("left_shoulder"),  g("left_hip"))
    angles["right_shoulder"] = angle_between(g("right_elbow"),    g("right_shoulder"), g("right_hip"))
    ms = mid(g("left_shoulder"), g("right_shoulder"))
    mh = mid(g("left_hip"),      g("right_hip"))
    mk = mid(g("left_knee"),     g("right_knee"))
    angles["torso"]     = angle_between(ms, mh, mk)
    angles["spine"]     = angle_between(g("nose"), ms, mh)
    angles["avg_knee"]  = (angles["left_knee"]  + angles["right_knee"])  / 2
    angles["avg_hip"]   = (angles["left_hip"]   + angles["right_hip"])   / 2
    angles["avg_elbow"] = (angles["left_elbow"] + angles["right_elbow"]) / 2
    return angles


def calculate_joint_angles_3d(lm: Dict) -> Dict[str, float]:
    """3D angle calculation using world landmarks (metric coords)."""
    if not lm:
        return {}
    g = lambda n: lm.get(n, (0., 0., 0.))
    angles: Dict[str, float] = {}
    angles["left_knee"]      = angle_between(g("left_hip"),       g("left_knee"),      g("left_ankle"),   use_3d=True)
    angles["right_knee"]     = angle_between(g("right_hip"),      g("right_knee"),     g("right_ankle"),  use_3d=True)
    angles["left_hip"]       = angle_between(g("left_shoulder"),  g("left_hip"),       g("left_knee"),    use_3d=True)
    angles["right_hip"]      = angle_between(g("right_shoulder"), g("right_hip"),      g("right_knee"),   use_3d=True)
    angles["left_elbow"]     = angle_between(g("left_shoulder"),  g("left_elbow"),     g("left_wrist"),   use_3d=True)
    angles["right_elbow"]    = angle_between(g("right_shoulder"), g("right_elbow"),    g("right_wrist"),  use_3d=True)
    angles["left_shoulder"]  = angle_between(g("left_elbow"),     g("left_shoulder"),  g("left_hip"),     use_3d=True)
    angles["right_shoulder"] = angle_between(g("right_elbow"),    g("right_shoulder"), g("right_hip"),    use_3d=True)
    ms = mid(g("left_shoulder"), g("right_shoulder"))
    mh = mid(g("left_hip"),      g("right_hip"))
    mk = mid(g("left_knee"),     g("right_knee"))
    angles["torso"]     = angle_between(ms, mh, mk, use_3d=True)
    angles["spine"]     = angle_between(g("nose"), ms, mh, use_3d=True)
    angles["avg_knee"]  = (angles["left_knee"]  + angles["right_knee"])  / 2
    angles["avg_hip"]   = (angles["left_hip"]   + angles["right_hip"])   / 2
    angles["avg_elbow"] = (angles["left_elbow"] + angles["right_elbow"]) / 2
    return angles


def compute_symmetry(angles: Dict) -> Dict:
    """
    Compare left vs right joint pairs.
    Returns score 0-100, worst joint name, and imbalance in degrees.
    """
    pairs = [
        ("left_knee",     "right_knee",     "knee"),
        ("left_elbow",    "right_elbow",    "elbow"),
        ("left_hip",      "right_hip",      "hip"),
        ("left_shoulder", "right_shoulder", "shoulder"),
    ]
    diffs = []
    worst_joint = "knee"
    worst_diff  = 0.0

    for l_name, r_name, label in pairs:
        l_val = angles.get(l_name, None)
        r_val = angles.get(r_name, None)
        if l_val is None or r_val is None:
            continue
        diff = abs(l_val - r_val)
        diffs.append(diff)
        if diff > worst_diff:
            worst_diff  = diff
            # Determine which side dominates
            side = "Left" if l_val > r_val else "Right"
            worst_joint = f"{side} {label}"

    if not diffs:
        return {"score": 100.0, "worst_joint": "—", "imbalance_deg": 0.0}

    mean_diff = float(np.mean(diffs))
    score     = max(0.0, round(100.0 - mean_diff, 1))
    return {
        "score":        score,
        "worst_joint":  worst_joint,
        "imbalance_deg": round(worst_diff, 1),
    }
