import pandas as pd
import os
from datetime import datetime, timedelta
from typing import List, Optional
from state_machine import RepRecord

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sessions.csv")
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

_COLS = ["date", "exercise", "rep_number", "form_score", "depth_score",
         "alignment_score", "stability_score", "duration_seconds",
         "eccentric_s", "concentric_s", "coaching_notes"]


def save_session(exercise: str, reps: List[RepRecord]):
    if not reps:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{
        "date":             now,
        "exercise":         exercise,
        "rep_number":       r.rep_number,
        "form_score":       round(r.form_score, 1),
        "depth_score":      round(r.depth_score, 1),
        "alignment_score":  round(r.alignment_score, 1),
        "stability_score":  round(r.stability_score, 1),
        "duration_seconds": round(r.duration_seconds, 2),
        "eccentric_s":      round(r.eccentric_s, 2),
        "concentric_s":     round(r.concentric_s, 2),
        "coaching_notes":   "; ".join(r.coaching_notes),
    } for r in reps]

    new_df = pd.DataFrame(rows, columns=_COLS)
    if os.path.exists(CSV_PATH):
        pd.concat([pd.read_csv(CSV_PATH), new_df], ignore_index=True).to_csv(CSV_PATH, index=False)
    else:
        new_df.to_csv(CSV_PATH, index=False)


def load_sessions() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=_COLS)
    return pd.read_csv(CSV_PATH)


def get_summary_stats(df: Optional[pd.DataFrame] = None) -> dict:
    if df is None:
        df = load_sessions()
    if df.empty:
        return {}
    return {
        "total_reps":     int(len(df)),
        "avg_form_score": round(df["form_score"].mean(), 1),
        "best_score":     round(df["form_score"].max(), 1),
        "exercises":      df["exercise"].value_counts().to_dict(),
        "sessions_count": df["date"].nunique(),
    }


def get_current_streak(df: pd.DataFrame) -> int:
    """Count consecutive calendar days that contain at least one session."""
    if df.empty:
        return 0
    df = df.copy()
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    unique_days = sorted(df["date_only"].unique(), reverse=True)

    today     = datetime.now().date()
    streak    = 0
    check_day = today

    for day in unique_days:
        if day == check_day or day == check_day - timedelta(days=1):
            streak    += 1
            check_day  = day
        elif day < check_day - timedelta(days=1):
            break

    return streak


def get_achievements(df: pd.DataFrame) -> List[dict]:
    """Return list of 5 achievement badge dicts."""
    achievements = [
        {"name": "7-day streak",  "icon": "🔥", "desc": "7 consecutive days",  "earned": False, "date": ""},
        {"name": "Perfect rep",   "icon": "⭐", "desc": "Score 100% on a rep", "earned": False, "date": ""},
        {"name": "Century",       "icon": "💯", "desc": "100+ total reps",      "earned": False, "date": ""},
        {"name": "Consistent",    "icon": "📅", "desc": "3 sessions in a week", "earned": False, "date": ""},
        {"name": "Form master",   "icon": "🏆", "desc": "Avg 90%+ over 10 reps","earned": False, "date": ""},
    ]

    if df.empty:
        return achievements

    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"])

    # 7-day streak
    if get_current_streak(df) >= 7:
        achievements[0]["earned"] = True
        achievements[0]["date"]   = df["date_parsed"].max().strftime("%Y-%m-%d")

    # Perfect rep
    perfect = df[df["form_score"] >= 99.9]
    if not perfect.empty:
        achievements[1]["earned"] = True
        achievements[1]["date"]   = pd.to_datetime(perfect["date"].iloc[0]).strftime("%Y-%m-%d")

    # Century
    if len(df) >= 100:
        achievements[2]["earned"] = True
        achievements[2]["date"]   = pd.to_datetime(df["date"].iloc[99]).strftime("%Y-%m-%d")

    # Consistent — 3 sessions (distinct dates) in any calendar week
    df["week"] = df["date_parsed"].dt.isocalendar().week.astype(int)
    df["year"] = df["date_parsed"].dt.year
    df["date_only"] = df["date_parsed"].dt.date
    weekly = df.groupby(["year", "week"])["date_only"].nunique()
    if (weekly >= 3).any():
        achievements[3]["earned"] = True
        achievements[3]["date"]   = df["date_parsed"].max().strftime("%Y-%m-%d")

    # Form master — avg >= 90 over last 10 reps
    if len(df) >= 10 and df["form_score"].tail(10).mean() >= 90:
        achievements[4]["earned"] = True
        achievements[4]["date"]   = df["date_parsed"].max().strftime("%Y-%m-%d")

    return achievements
