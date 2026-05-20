import pandas as pd
from typing import List, Dict
from personalization import UserProfile, get_rep_target


_PLAN_TEMPLATE = [
    {"day": "Mon", "exercises": ["Squat", "Shoulder Press"]},
    {"day": "Tue", "exercises": []},
    {"day": "Wed", "exercises": ["Push-up", "Squat"]},
    {"day": "Thu", "exercises": []},
    {"day": "Fri", "exercises": ["Squat", "Push-up", "Shoulder Press"]},
]


def _analyse_exercise(df: pd.DataFrame, exercise: str) -> Dict:
    """Return last-3-session avg score and trend for one exercise."""
    ex_df = df[df["exercise"] == exercise]
    if ex_df.empty:
        return {"avg_score": None, "sessions": 0}
    # Group by date, compute avg score per session
    ex_df = ex_df.copy()
    ex_df["date_only"] = pd.to_datetime(ex_df["date"]).dt.date
    session_avgs = ex_df.groupby("date_only")["form_score"].mean().sort_index()
    last_3 = session_avgs.tail(3)
    return {
        "avg_score": round(last_3.mean(), 1) if not last_3.empty else None,
        "sessions":  len(session_avgs),
    }


def generate_weekly_plan(profile: UserProfile, df: pd.DataFrame) -> List[Dict]:
    """
    Build a 5-day plan with progressive overload.
    Returns list of day dicts:
      [{"day": "Mon", "rest": False, "exercises": [{"name", "reps", "note"}]}]
    """
    plan = []

    for template_day in _PLAN_TEMPLATE:
        day_name = template_day["day"]

        if not template_day["exercises"]:
            plan.append({"day": day_name, "rest": True, "exercises": []})
            continue

        day_exercises = []
        for ex_name in template_day["exercises"]:
            base_reps = get_rep_target(ex_name, profile)
            note = ""

            if not df.empty:
                stats = _analyse_exercise(df, ex_name)
                avg   = stats["avg_score"]
                if avg is not None:
                    if avg > 80 and stats["sessions"] >= 3:
                        base_reps += 2
                        note = "Progressive overload — +2 reps"
                    elif avg < 55:
                        base_reps = max(1, base_reps - 1)
                        note = "Focus on form — reduce weight"

            day_exercises.append({
                "name": ex_name,
                "reps": base_reps,
                "note": note,
            })

        plan.append({"day": day_name, "rest": False, "exercises": day_exercises})

    return plan
