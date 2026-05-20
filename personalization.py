from dataclasses import dataclass

@dataclass
class UserProfile:
    name: str = "Athlete"
    age: int = 30
    weight_kg: float = 70.0
    fitness_level: str = "Intermediate"
    goal: str = "General fitness"

_REP_TARGETS = {
    "Squat":          {"Beginner": 8,  "Intermediate": 12, "Advanced": 20},
    "Push-up":        {"Beginner": 5,  "Intermediate": 10, "Advanced": 20},
    "Shoulder Press": {"Beginner": 8,  "Intermediate": 12, "Advanced": 16},
}
_STRICTNESS = {"Beginner": 1.4, "Intermediate": 1.0, "Advanced": 0.7}
_REST       = {"Beginner": 90,  "Intermediate": 60,  "Advanced": 30}

def get_rep_target(exercise, profile):
    d    = _REP_TARGETS.get(exercise, {"Beginner": 8, "Intermediate": 12, "Advanced": 16})
    base = d.get(profile.fitness_level, 12)
    if profile.goal == "Weight loss":  base = int(base * 1.2)
    if profile.goal == "Muscle gain":  base = int(base * 0.85)
    return max(1, base)

def get_angle_tolerance(base, profile):
    return base * _STRICTNESS.get(profile.fitness_level, 1.0)

def get_rest_suggestion(profile):
    return _REST.get(profile.fitness_level, 60)

def get_intensity_label(profile):
    m = {
        ("Beginner",     "Weight loss"):    "Light cardio circuits",
        ("Beginner",     "Muscle gain"):    "Foundation strength",
        ("Beginner",     "General fitness"):"Full-body basics",
        ("Intermediate", "Weight loss"):    "Moderate HIIT intervals",
        ("Intermediate", "Muscle gain"):    "Progressive overload",
        ("Intermediate", "General fitness"):"Balanced conditioning",
        ("Advanced",     "Weight loss"):    "High-intensity metabolic",
        ("Advanced",     "Muscle gain"):    "Heavy compound lifts",
        ("Advanced",     "General fitness"):"Athletic performance",
    }
    return m.get((profile.fitness_level, profile.goal), "General workout")

def calorie_estimate(exercise, reps, weight_kg, duration_seconds):
    met = {"Squat": 5.0, "Push-up": 4.0, "Shoulder Press": 4.5}.get(exercise, 4.0)
    return round(met * weight_kg * (duration_seconds / 3600.0), 2)
