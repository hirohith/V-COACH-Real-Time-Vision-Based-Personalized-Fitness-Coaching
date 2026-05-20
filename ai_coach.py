import os
import json
import requests
from typing import List
from state_machine import RepRecord
from personalization import UserProfile

GEMINI_API_URL = GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def generate_session_summary(
    reps: List[RepRecord],
    exercise: str,
    profile: UserProfile,
    api_key: str = "",
) -> str:
    """
    Calls Google Gemini API (free tier — 1500 req/day on gemini-1.5-flash).
    Returns empty string if API key is missing or call fails.
    """
    key = api_key or os.environ.get(
        "GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not key or not reps:
        return ""

    scores = [r.form_score for r in reps]
    eccentrics = [r.eccentric_s for r in reps if r.eccentric_s > 0]
    best_rep = max(reps, key=lambda r: r.form_score)
    worst_rep = min(reps, key=lambda r: r.form_score)

    payload_data = {
        "exercise":        exercise,
        "total_reps":      len(reps),
        "avg_form_score":  round(sum(scores) / len(scores), 1),
        "best_rep":        {"number": best_rep.rep_number,  "score": best_rep.form_score},
        "worst_rep":       {"number": worst_rep.rep_number, "score": worst_rep.form_score,
                            "notes": worst_rep.coaching_notes},
        "avg_eccentric_s": round(sum(eccentrics) / len(eccentrics), 2) if eccentrics else None,
        "tempo_trend":     [r.eccentric_s for r in reps],
        "profile": {
            "level": profile.fitness_level,
            "goal":  profile.goal,
            "age":   profile.age,
        },
    }

    prompt = (
        "You are a personal fitness coach. Be concise, specific, and encouraging. "
        "Give exactly 3 coaching points in numbered list format. "
        "Each point must be actionable and tied directly to the data.\n\n"
        f"Review this workout session:\n{json.dumps(payload_data, indent=2)}"
    )

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 400, "temperature": 0.7}
    }

    try:
        resp = requests.post(
            f"{GEMINI_API_URL}?key={key}", json=body, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.Timeout:
        return "AI coaching timed out — check your internet connection."
    except requests.exceptions.HTTPError:
        if resp.status_code == 400:
            return "Invalid Gemini API key — get one free at aistudio.google.com."
        if resp.status_code == 404:
            return "Gemini model not found — check your API key has access at aistudio.google.com."
        if resp.status_code == 429:
            return "Rate limit reached — wait 60 seconds and try again. (Free tier: 15 requests/min)"
        return f"API error {resp.status_code}."
    except (KeyError, IndexError):
        return "Unexpected response from Gemini API."
    except Exception as e:
        return f"Could not reach AI coach: {e}"
