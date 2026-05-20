from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FormScore:
    depth_score: float
    alignment_score: float
    stability_score: float
    total: float
    grade: str
    suggestions: List[str] = field(default_factory=list)

    @classmethod
    def from_parts(cls, depth, alignment, stability, suggestions=None):
        total = round(min(max(depth + alignment + stability, 0), 100), 1)
        grade = "A" if total>=90 else "B" if total>=75 else "C" if total>=60 else "D" if total>=40 else "F"
        return cls(round(depth,1), round(alignment,1), round(stability,1), total, grade, suggestions or [])

    def as_dict(self):
        return {"Depth": self.depth_score, "Alignment": self.alignment_score, "Stability": self.stability_score}


def score_depth(achieved, ideal_min, ideal_max, max_pts=40.0):
    if ideal_min <= achieved <= ideal_max:
        return max_pts
    if achieved < ideal_min:
        return round(max_pts * (1.0 - min((ideal_min - achieved) / ideal_min, 1.0) * 0.8), 1)
    return round(max_pts * (1.0 - min((achieved - ideal_max) / 90.0, 1.0) * 0.5), 1)


def score_alignment(angles, rules, max_pts=30.0):
    if not rules:
        return max_pts
    tw = sum(r["weight"] for r in rules)
    earned = 0.0
    for r in rules:
        diff = abs(angles.get(r["angle"], r["ideal"]) - r["ideal"])
        earned += (r["weight"] / tw) * max_pts * max(0.0, 1.0 - diff / (r["tolerance"] * 2))
    return round(earned, 1)
