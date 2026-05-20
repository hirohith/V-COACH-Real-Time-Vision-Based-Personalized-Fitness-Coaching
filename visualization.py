import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

CYAN   = (255, 200, 0)
GREEN  = (0, 230, 100)
RED    = (0, 60, 255)
ORANGE = (0, 140, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DARK   = (20, 20, 20)
YELLOW = (0, 220, 220)
BLUE   = (255, 120, 0)


def _shadow(frame, text, pos, scale=0.6, color=WHITE, thickness=1):
    x, y = pos
    cv2.putText(frame, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,     thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y),   cv2.FONT_HERSHEY_SIMPLEX, scale, color,     thickness,   cv2.LINE_AA)


def _semi_rect(frame, x0, y0, x1, y1, alpha=0.65):
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), DARK, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def draw_angle_labels(frame, lm_dict, angles):
    h, w = frame.shape[:2]
    joint_map = {
        "left_knee":   "left_knee",
        "right_knee":  "right_knee",
        "left_elbow":  "left_elbow",
        "right_elbow": "right_elbow",
        "left_hip":    "left_hip",
        "right_hip":   "right_hip",
    }
    for aname, lname in joint_map.items():
        if aname in angles and lname in lm_dict:
            x, y, _ = lm_dict[lname]
            _shadow(frame, f"{int(angles[aname])}°", (int(x*w)+10, int(y*h)-10), scale=0.5, color=CYAN)
    return frame


def draw_rep_counter(frame, reps, target, name, state=""):
    h, w = frame.shape[:2]
    _semi_rect(frame, 10, 10, 230, 115)
    cv2.putText(frame, name.upper(), (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN,  2, cv2.LINE_AA)
    cv2.putText(frame, f"{reps} / {target}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                GREEN if reps < target else YELLOW, 3, cv2.LINE_AA)
    if state:
        cv2.putText(frame, state, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
    return frame


def draw_rep_counter_dual(frame, reps_a, reps_b, target, name):
    """HUD for multi-person mode."""
    h, w = frame.shape[:2]
    _semi_rect(frame, 10, 10, 270, 85)
    cv2.putText(frame, name.upper(), (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN,   1, cv2.LINE_AA)
    cv2.putText(frame, f"A: {reps_a}/{target}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"B: {reps_b}/{target}", (150, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
    return frame


def draw_coaching_message(frame, message, severity="info"):
    h, w = frame.shape[:2]
    color = {"info": GREEN, "warn": ORANGE, "error": RED}.get(severity, GREEN)
    ts, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x = (w - ts[0]) // 2; y = h - 25; pad = 12
    _semi_rect(frame, x-pad, y-ts[1]-pad, x+ts[0]+pad, y+pad, alpha=0.70)
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame


def draw_form_score(frame, score, breakdown=None):
    h, w = frame.shape[:2]
    pw, ph = 180, 100 if not breakdown else 140
    x0 = w - pw - 10
    _semi_rect(frame, x0, 10, w-10, 10+ph)
    lc = GREEN if score >= 75 else ORANGE if score >= 50 else RED
    cv2.putText(frame, "FORM SCORE", (x0+8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(score)}%", (x0+8, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2, lc, 3, cv2.LINE_AA)
    bx, by, bw = x0+8, 72, pw-16
    cv2.rectangle(frame, (bx, by), (bx+bw, by+8), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(bw*score/100), by+8), lc, -1)
    if breakdown:
        oy = 88
        for k, v in breakdown.items():
            cv2.putText(frame, f"{k[:8]}: {int(v)}", (x0+8, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)
            oy += 16
    return frame


def draw_calorie_counter(frame, calories):
    h, w = frame.shape[:2]
    _shadow(frame, f"~{calories:.1f} kcal", (14, h-55), scale=0.55, color=YELLOW)
    return frame


def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    _shadow(frame, f"FPS:{fps:.0f}", (w-80, h-12), scale=0.45, color=WHITE)
    return frame


def draw_tempo_hud(frame, eccentric_s: float, concentric_s: float):
    """
    Bottom-right tempo display.
    Eccentric colour: green 2-4s, orange <2s, red <1s.
    """
    h, w = frame.shape[:2]
    if eccentric_s <= 0 and concentric_s <= 0:
        return frame

    if eccentric_s < 1.0:
        ecc_color = RED
    elif eccentric_s < 2.0:
        ecc_color = ORANGE
    else:
        ecc_color = GREEN

    panel_w, panel_h = 190, 50
    x0 = w - panel_w - 10
    y0 = h - panel_h - 65
    _semi_rect(frame, x0, y0, x0+panel_w, y0+panel_h)

    cv2.putText(frame, "TEMPO", (x0+8, y0+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"DOWN {eccentric_s:.1f}s", (x0+8, y0+38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ecc_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"UP {concentric_s:.1f}s", (x0+105, y0+38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, CYAN, 1, cv2.LINE_AA)
    return frame


def draw_symmetry_badge(frame, sym: dict):
    """
    Bottom-left symmetry score badge.
    """
    if not sym:
        return frame
    h, w = frame.shape[:2]
    score      = sym.get("score", 100.0)
    worst      = sym.get("worst_joint", "")
    imbalance  = sym.get("imbalance_deg", 0.0)

    color = GREEN if score >= 85 else ORANGE if score >= 70 else RED

    _semi_rect(frame, 10, h-115, 220, h-60)
    cv2.putText(frame, f"SYM {int(score)}%", (18, h-90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if imbalance > 15 and worst:
        cv2.putText(frame, f"{worst} dominant", (18, h-68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ORANGE, 1, cv2.LINE_AA)
    return frame


def draw_warmup_overlay(frame, stretch_name: str, cue: str, seconds_remaining: float, progress: float):
    """Full-screen warmup countdown overlay."""
    h, w = frame.shape[:2]
    _semi_rect(frame, 0, h//2 - 70, w, h//2 + 80, alpha=0.75)
    cv2.putText(frame, "WARM-UP", (w//2 - 70, h//2 - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, stretch_name.upper(), (w//2 - len(stretch_name)*8, h//2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, cue, (w//2 - len(cue)*5, h//2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(seconds_remaining)}s", (w//2 - 22, h//2 + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3, cv2.LINE_AA)
    # Progress bar
    bar_x, bar_y, bar_w = 40, h//2 + 72, w - 80
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+8), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*progress), bar_y+8), GREEN, -1)
    return frame
