"""
VCoach v2 – AI Home Fitness Trainer (fully upgraded)
Run:  streamlit run app.py   (from inside the VCoach/ folder)
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
import time
import os
from itertools import cycle

from pose_detector import PoseDetector
from angle_calculator import calculate_joint_angles, calculate_joint_angles_3d, compute_symmetry
from visualization import (draw_angle_labels, draw_rep_counter, draw_rep_counter_dual,
                           draw_coaching_message, draw_form_score, draw_calorie_counter,
                           draw_fps, draw_tempo_hud, draw_symmetry_badge, draw_warmup_overlay)
from personalization import (UserProfile, get_rep_target, get_intensity_label,
                             get_rest_suggestion, calorie_estimate)
from squat import SquatExercise
from pushup import PushupExercise
from shoulder_press import ShoulderPressExercise
from session_tracker import (save_session, load_sessions, get_summary_stats,
                             get_current_streak, get_achievements)
from exercise_classifier import ExerciseClassifier
from video_exporter import RepVideoExporter
from workout_planner import generate_weekly_plan
from ai_coach import generate_session_summary
from warmup import WarmupRoutine

try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty("rate", 160)
    TTS_OK = True
except Exception:
    TTS_OK = False

# ── Dirs ──────────────────────────────────────────────────────────────────
CLIPS_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data", "rep_clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="VCoach v2", page_icon="🏋️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
h1,h2,h3{font-family:'Rajdhani',sans-serif!important;}
.stApp{background:#0d0f14;color:#e8eaf0;}
section[data-testid="stSidebar"]{background:#13161e;border-right:1px solid #1e2230;}
[data-testid="stMetric"]{background:#1a1e2a;border:1px solid #252a3a;border-radius:12px;padding:12px 16px;}
[data-testid="stMetricLabel"]{color:#7c8099;font-size:.75rem;text-transform:uppercase;}
[data-testid="stMetricValue"]{color:#00e5ff;font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;}
.stButton>button{background:linear-gradient(135deg,#00b4d8,#0077b6);color:#fff;border:none;
  border-radius:8px;font-family:'Rajdhani',sans-serif;font-weight:600;font-size:1rem;
  padding:10px 24px;width:100%;transition:all .2s;}
.stButton>button:hover{background:linear-gradient(135deg,#48cae4,#0096c7);transform:translateY(-1px);}
.stop-btn>button{background:linear-gradient(135deg,#e63946,#c1121f)!important;}
.badge-earned{display:inline-block;padding:8px 14px;border-radius:12px;background:#1a2a3a;
  border:1px solid #00b4d8;text-align:center;margin:4px;}
.badge-locked{display:inline-block;padding:8px 14px;border-radius:12px;background:#13161e;
  border:1px solid #252a3a;text-align:center;margin:4px;opacity:0.45;}
.fatigue-banner{background:#3a1020;border:1px solid #e63946;border-radius:8px;
  padding:10px 16px;color:#ff6b6b;font-weight:500;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
_DEFAULTS = {
    "running": False, "exercise_obj": None, "pose_detector": None, "cap": None,
    "exercise_obj_b": None,
    "session_reps": [], "current_exercise": "Squat", "calories": 0.0,
    "session_start_ts": None, "last_coaching_msg": "", "last_score": 0.0,
    "last_breakdown": {}, "voice_enabled": False,
    "last_eccentric": 0.0, "last_concentric": 0.0,
    "fatigue_warning": "", "ai_summary": "",
    "warmup_done": False, "warmup_start_ts": None, "wu_cap": None,
    "video_exporter": None, "classifier": None,
    "auto_detected_ex": "", "auto_conf": 0.0,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏋️ VCoach v2")
    st.caption("AI-Powered Home Fitness Trainer")
    st.divider()

    st.markdown("### 👤 Profile")
    user_name = st.text_input("Name",   value="Athlete")
    user_age = st.slider("Age", 18, 75, 28)
    user_weight = st.number_input("Weight (kg)", 40.0, 200.0, 70.0, step=0.5)
    fitness_level = st.selectbox(
        "Fitness Level", ["Beginner", "Intermediate", "Advanced"], index=1)
    goal = st.selectbox(
        "Goal", ["General fitness", "Weight loss", "Muscle gain"])
    profile = UserProfile(user_name, user_age,
                          user_weight, fitness_level, goal)

    st.info(
        f"🎯 {get_intensity_label(profile)}  \n⏱ Rest: {get_rest_suggestion(profile)}s")
    st.divider()

    st.markdown("### 🏃 Exercise")
    exercise_choices = ["Squat", "Push-up", "Shoulder Press", "Auto-detect"]
    exercise_name = st.selectbox("Choose exercise", exercise_choices)

    target_reps = get_rep_target(
        exercise_name if exercise_name != "Auto-detect" else "Squat", profile
    )
    st.caption(f"Target reps: **{target_reps}**")
    st.divider()

    warmup_enabled = st.toggle("🔥 Warm-up first", value=False)
    multi_person = st.toggle("👥 Multi-person mode", value=False)
    if TTS_OK:
        st.session_state["voice_enabled"] = st.toggle(
            "🔊 Voice Coach", value=False)

    st.markdown("### 🔑 AI Coach")
    api_key = st.text_input("GEMINI_API_KEY (optional)",
                            type="password", value="")

    st.markdown("### 📷 Camera")
    cam_index = st.number_input("Camera index", 0, 5, 0, step=1)

# ── Main tabs ─────────────────────────────────────────────────────────────
st.markdown("# VCoach v2  <span style='font-size:1rem;color:#7c8099;font-weight:400;'>AI Fitness Trainer</span>",
            unsafe_allow_html=True)
tab_workout, tab_dashboard = st.tabs(["🏋️ Workout", "📊 Dashboard"])

# ═════════════════════════════ WORKOUT TAB ════════════════════════════════
with tab_workout:

    # Fatigue banner (persistent when set)
    if st.session_state["fatigue_warning"]:
        st.markdown(
            f'<div class="fatigue-banner">⚠️ {st.session_state["fatigue_warning"]}</div>',
            unsafe_allow_html=True
        )

    # AI summary from previous session
    if st.session_state["ai_summary"]:
        st.info(f"**Your AI coach says:**\n\n{st.session_state['ai_summary']}")
        if st.button("Clear coaching summary"):
            st.session_state["ai_summary"] = ""
            st.rerun()

    c1, c2, c3 = st.columns([2, 2, 3])

    def _build_exercise(name, prof, reps):
        mapping = {"Squat": SquatExercise, "Push-up": PushupExercise,
                   "Shoulder Press": ShoulderPressExercise}
        cls = mapping.get(name, SquatExercise)
        return cls(profile=prof, target_reps=reps)

    with c1:
        if not st.session_state["running"]:
            start_disabled = warmup_enabled and not st.session_state["warmup_done"]
            if st.button("▶  Start Workout", use_container_width=True, disabled=start_disabled):
                ex_name = exercise_name if exercise_name != "Auto-detect" else "Squat"
                st.session_state.update({
                    "exercise_obj":      _build_exercise(ex_name, profile, target_reps),
                    "exercise_obj_b":    _build_exercise(ex_name, profile, target_reps) if multi_person else None,
                    "pose_detector":     PoseDetector(),
                    "cap":               cv2.VideoCapture(int(cam_index)),
                    "running":           True,
                    "session_reps":      [],
                    "calories":          0.0,
                    "current_exercise":  ex_name,
                    "session_start_ts":  time.time(),
                    "last_coaching_msg": "",
                    "fatigue_warning":   "",
                    "ai_summary":        "",
                    "last_eccentric":    0.0,
                    "last_concentric":   0.0,
                    "video_exporter":    RepVideoExporter(CLIPS_DIR),
                    "classifier":        ExerciseClassifier() if exercise_name == "Auto-detect" else None,
                    "auto_detected_ex":  "",
                    "auto_conf":         0.0,
                })
                cap = st.session_state["cap"]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                st.rerun()

    with c2:
        if st.session_state["running"]:
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            if st.button("⏹  Stop & Save", use_container_width=True):
                ex = st.session_state["exercise_obj"]
                reps = ex.rep_history if ex else []
                exname = st.session_state["current_exercise"]
                save_session(exname, reps)

                ve = st.session_state.get("video_exporter")
                if ve:
                    ve.close()

                if st.session_state["cap"]:
                    st.session_state["cap"].release()
                if st.session_state["pose_detector"]:
                    st.session_state["pose_detector"].release()
                if st.session_state.get("wu_cap"):
                    st.session_state["wu_cap"].release()
                    st.session_state["wu_cap"] = None

                # AI coaching summary
                if reps and api_key:
                    with st.spinner("Getting AI coaching feedback..."):
                        summary = generate_session_summary(
                            reps, exname, profile, api_key)
                    st.session_state["ai_summary"] = summary

                st.session_state.update({
                    "running": False, "exercise_obj": None,
                    "cap": None, "warmup_done": False,
                })
                st.success(f"✅ Session saved — {len(reps)} reps logged!")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        if st.session_state["running"]:
            elapsed = time.time() - \
                (st.session_state["session_start_ts"] or time.time())
            m, s = divmod(int(elapsed), 60)
            st.markdown(
                f"<div style='text-align:right;padding-top:8px;'>"
                f"<span style='font-family:Rajdhani;font-size:1.4rem;color:#00e5ff;'>⏱ {m:02d}:{s:02d}</span>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Warm-up phase (with live camera) ──────────────────────────────────
    if warmup_enabled and not st.session_state["warmup_done"] and not st.session_state["running"]:
        st.markdown("### 🔥 Warm-up")
        warmup = WarmupRoutine()

        if st.session_state["warmup_start_ts"] is None:
            if st.button("Start warm-up", use_container_width=True):
                st.session_state["warmup_start_ts"] = time.time()
                st.rerun()
        else:
            wu_feed_col, wu_stats_col = st.columns([3, 1])
            wu_frame_ph = wu_feed_col.empty()

            with wu_stats_col:
                st.markdown("#### Warm-up")
                wu_prog_ph = st.empty()
                wu_name_ph = st.empty()
                wu_cue_ph = st.empty()
                wu_timer_ph = st.empty()
                wu_stretch_ph = st.empty()
                st.divider()
                if st.button("Skip warm-up", use_container_width=True):
                    st.session_state["warmup_done"] = True
                    st.session_state["warmup_start_ts"] = None
                    st.rerun()

            # Open camera once and store in session_state
            if st.session_state["wu_cap"] is None:
                cap_wu = cv2.VideoCapture(int(cam_index))
                cap_wu.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap_wu.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.session_state["wu_cap"] = cap_wu

            cap_wu = st.session_state["wu_cap"]
            elapsed_wu = time.time() - st.session_state["warmup_start_ts"]
            info = warmup.tick(elapsed_wu)

            if info is None:
                cap_wu.release()
                st.session_state["wu_cap"] = None
                st.session_state["warmup_done"] = True
                st.session_state["warmup_start_ts"] = None
                st.success("Warm-up complete! Press Start Workout above.")
                st.rerun()
            else:
                ret, wu_frame = cap_wu.read()
                if ret:
                    wu_frame = cv2.flip(wu_frame, 1)
                    draw_warmup_overlay(
                        wu_frame,
                        info["name"],
                        info["cue"],
                        info["seconds_remaining"],
                        info["overall_progress"],
                    )
                    wu_frame_ph.image(
                        cv2.cvtColor(wu_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                else:
                    wu_frame_ph.warning(
                        "Camera not accessible. Check camera index.")

                wu_prog_ph.progress(
                    info["overall_progress"],
                    text=f"Overall: {int(info['overall_progress']*100)}%"
                )
                wu_name_ph.markdown(f"**{info['name']}**")
                wu_cue_ph.caption(info["cue"])
                wu_timer_ph.metric(
                    "Time remaining", f"{int(info['seconds_remaining'])}s")
                wu_stretch_ph.caption(
                    f"Stretch {info['stretch_index']+1} of {info['stretch_count']}"
                )
                time.sleep(0.033)
                st.rerun()

    # ── Live workout ───────────────────────────────────────────────────────
    if st.session_state["running"]:
        feed_col, stats_col = st.columns([3, 1])
        frame_ph = feed_col.empty()
        coach_ph = feed_col.empty()

        with stats_col:
            st.markdown("#### Live Stats")
            reps_ph = st.empty()
            score_ph = st.empty()
            state_ph = st.empty()
            sym_ph = st.empty()
            st.divider()
            st.markdown("#### Last Rep")
            tips_ph = st.empty()
            tempo_ph = st.empty()
            st.divider()
            cal_ph = st.empty()
            if exercise_name == "Auto-detect":
                detect_ph = st.empty()

        cap = st.session_state["cap"]
        detector = st.session_state["pose_detector"]
        exercise = st.session_state["exercise_obj"]
        exercise_b = st.session_state.get("exercise_obj_b")
        exporter = st.session_state.get("video_exporter")
        classifier = st.session_state.get("classifier")

        fps_timer = time.time()
        fps_val = 0.0
        frame_count = 0

        while st.session_state["running"]:
            ret, frame = cap.read()
            if not ret:
                coach_ph.warning(
                    "⚠️ Camera not accessible. Check camera index in sidebar.")
                time.sleep(0.5)
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            angles = {}
            angles_b = {}
            bad_joints = []
            bad_joints_b = []
            live_score = None

            if not multi_person:
                # ── Single person ──────────────────────────────────────
                pf = detector.process_frame(frame)
                if pf.landmarks:
                    # Prefer 3D angles when world landmarks available
                    if pf.world_landmark_dict:
                        angles = calculate_joint_angles_3d(
                            pf.world_landmark_dict)
                        if not angles:
                            angles = calculate_joint_angles(pf.landmark_dict)
                    else:
                        angles = calculate_joint_angles(pf.landmark_dict)

                    sym = compute_symmetry(angles)
                    bad_joints = exercise.get_bad_joints(angles)
                    live_score = exercise.get_form_score_live(angles)

                    # Auto-detection
                    if classifier is not None:
                        ready, det_name, det_conf = classifier.is_confident(
                            angles)
                        st.session_state["auto_conf"] = det_conf
                        if ready and det_name != st.session_state["auto_detected_ex"]:
                            st.session_state["auto_detected_ex"] = det_name
                            st.session_state["current_exercise"] = det_name
                            st.session_state["exercise_obj"] = _build_exercise(
                                det_name, profile, target_reps)
                            exercise = st.session_state["exercise_obj"]

                    completed = exercise.update(
                        angles, pf, confidence=pf.confidence)

                    if completed:
                        elapsed_total = time.time() - \
                            st.session_state["session_start_ts"]
                        st.session_state["calories"] = calorie_estimate(
                            st.session_state["current_exercise"],
                            exercise.reps, profile.weight_kg, elapsed_total)
                        st.session_state["last_score"] = completed.form_score
                        st.session_state["last_breakdown"] = completed.form_score and {
                            "Depth":     completed.depth_score,
                            "Alignment": completed.alignment_score,
                            "Stability": completed.stability_score,
                        }
                        st.session_state["last_eccentric"] = completed.eccentric_s
                        st.session_state["last_concentric"] = completed.concentric_s

                        if exporter:
                            exporter.on_rep_complete(
                                completed.rep_number, completed.form_score)

                        fatigue_msg = exercise.check_fatigue()
                        if fatigue_msg:
                            st.session_state["fatigue_warning"] = fatigue_msg

                        if completed.coaching_notes:
                            tips_ph.markdown(
                                "".join(f'<span style="display:inline-block;background:#1a2a3a;border:1px solid #0077b6;'
                                        f'border-radius:20px;padding:4px 12px;font-size:.82rem;color:#90e0ef;margin:3px;">'
                                        f'💡 {t}</span>'
                                        for t in completed.coaching_notes),
                                unsafe_allow_html=True)
                        tempo_ph.caption(
                            f"Eccentric {completed.eccentric_s:.1f}s | Concentric {completed.concentric_s:.1f}s")

                    annotated = detector.draw_landmarks(
                        pf, highlight_joints=bad_joints)
                    if exporter:
                        exporter.add_frame(annotated)
                    draw_rep_counter(annotated, exercise.reps, target_reps,
                                     st.session_state["current_exercise"], state=exercise.state.value)
                else:
                    annotated = frame.copy()
                    sym = {}
                    draw_rep_counter(annotated, exercise.reps, target_reps,
                                     st.session_state["current_exercise"])

            else:
                # ── Dual person ────────────────────────────────────────
                pf_a, pf_b = detector.process_frame_dual(frame)
                annotated = frame.copy()

                if pf_a.landmarks:
                    angles = calculate_joint_angles(pf_a.landmark_dict)
                    bad_joints = exercise.get_bad_joints(angles)
                    live_score = exercise.get_form_score_live(angles)
                    exercise.update(angles, pf_a, confidence=pf_a.confidence)
                    detector.draw_landmarks_on(annotated, pf_a,
                                               dot_color=(255, 200, 0),
                                               highlight_joints=bad_joints)

                if pf_b.landmarks and exercise_b:
                    angles_b = calculate_joint_angles(pf_b.landmark_dict)
                    bad_joints_b = exercise_b.get_bad_joints(angles_b)
                    exercise_b.update(
                        angles_b, pf_b, confidence=pf_b.confidence)
                    detector.draw_landmarks_on(annotated, pf_b,
                                               dot_color=(0, 165, 255),
                                               highlight_joints=bad_joints_b)

                sym = {}
                reps_b = exercise_b.reps if exercise_b else 0
                draw_rep_counter_dual(annotated, exercise.reps, reps_b, target_reps,
                                      st.session_state["current_exercise"])

            # ── Common overlays ────────────────────────────────────────
            score_val = live_score.total if live_score else st.session_state["last_score"]
            score_brk = live_score.as_dict(
            ) if live_score else st.session_state["last_breakdown"]

            draw_form_score(annotated, score_val, breakdown=score_brk)
            if angles:
                draw_angle_labels(
                    annotated, pf.landmark_dict if not multi_person else {}, angles)
            draw_calorie_counter(annotated, st.session_state["calories"])
            draw_symmetry_badge(annotated, sym)
            draw_tempo_hud(annotated, st.session_state["last_eccentric"],
                           st.session_state["last_concentric"])

            if frame_count % 15 == 0:
                fps_val = 15 / max(time.time() - fps_timer, 0.001)
                fps_timer = time.time()
            draw_fps(annotated, fps_val)

            msg = exercise.pop_coaching_message()
            if msg:
                st.session_state["last_coaching_msg"] = msg
                if st.session_state["voice_enabled"] and TTS_OK:
                    try:
                        _tts.say(msg)
                        _tts.runAndWait()
                    except Exception:
                        pass

            if st.session_state["last_coaching_msg"]:
                sev = "warn" if bad_joints else "info"
                if st.session_state["fatigue_warning"]:
                    sev = "error"
                draw_coaching_message(
                    annotated, st.session_state["last_coaching_msg"], sev)

            frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

            reps_ph.metric("Reps",  f"{exercise.reps} / {target_reps}")
            score_ph.metric("Form", f"{int(score_val)}%")
            sc = {"IDLE": "#7c8099", "START": "#4cff91", "DOWN": "#ff9800", "UP": "#00e5ff"}.get(
                exercise.state.value, "#fff")
            state_ph.markdown(
                f'<span style="display:inline-block;padding:3px 12px;border-radius:12px;'
                f'font-size:.75rem;font-weight:600;background:{sc}22;color:{sc};border:1px solid {sc}44;">'
                f'{exercise.state.value}</span>', unsafe_allow_html=True)
            if sym:
                sym_ph.metric("Symmetry", f"{int(sym.get('score', 100))}%")
            cal_ph.metric(
                "Calories", f"~{st.session_state['calories']:.1f} kcal")

            if exercise_name == "Auto-detect" and classifier:
                detect_ph.caption(
                    f"Detected: {st.session_state['auto_detected_ex'] or '...'} "
                    f"({int(st.session_state['auto_conf']*100)}%)")

            if exercise.reps >= target_reps:
                coach_ph.success(
                    f"🎉 {target_reps} reps complete! Rest {get_rest_suggestion(profile)}s then continue.")

            time.sleep(0.01)

    elif not st.session_state["running"]:
        if not (warmup_enabled and not st.session_state["warmup_done"]):
            st.markdown("""
<div style="background:#13161e;border:2px dashed #252a3a;border-radius:16px;
            padding:60px 40px;text-align:center;color:#7c8099;">
  <div style="font-size:3.5rem;">📷</div>
  <h3 style="font-family:Rajdhani;color:#e8eaf0;margin-top:12px;">Camera Ready</h3>
  <p>Configure your profile in the sidebar, choose an exercise, then press <strong>▶ Start Workout</strong></p>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════ DASHBOARD TAB ══════════════════════════════
with tab_dashboard:
    st.markdown("### 📊 Performance Dashboard")
    df = load_sessions()

    if df.empty:
        st.info("No session data yet. Complete a workout to see your stats here.")
    else:
        df["date"] = pd.to_datetime(df["date"])
        s = get_summary_stats(df)

        # ── Top metrics ────────────────────────────────────────────────
        streak = get_current_streak(df)
        m0, m1, m2, m3, m4 = st.columns(5)
        m0.metric("🔥 Streak",       f"{streak} days")
        m1.metric("Total Reps",      s.get("total_reps", 0))
        m2.metric("Avg Form Score",  f"{s.get('avg_form_score', 0)}%")
        m3.metric("Best Rep Score",  f"{s.get('best_score', 0)}%")
        m4.metric("Sessions Logged", s.get("sessions_count", 0))

        # ── Achievements ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🏅 Achievements")
        badges = get_achievements(df)
        badge_cols = st.columns(len(badges))
        for col, badge in zip(badge_cols, badges):
            css_class = "badge-earned" if badge["earned"] else "badge-locked"
            date_str = f"<br><small>{badge['date']}</small>" if badge["earned"] and badge["date"] else ""
            col.markdown(
                f'<div class="{css_class}">'
                f'<span style="font-size:22px;">{badge["icon"]}</span><br>'
                f'<strong>{badge["name"]}</strong><br>'
                f'<small style="color:#7c8099;">{badge["desc"]}</small>'
                f'{date_str}</div>',
                unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────────
        st.markdown("---")
        cc, hc = st.columns(2)
        with cc:
            st.markdown("#### Form Score Trend")
            td = df.groupby(df["date"].dt.date)[
                "form_score"].mean().reset_index()
            td.columns = ["Date", "Avg Form Score"]
            st.line_chart(td.set_index("Date"))
        with hc:
            st.markdown("#### Reps by Exercise")
            ec = df["exercise"].value_counts().reset_index()
            ec.columns = ["Exercise", "Reps"]
            st.bar_chart(ec.set_index("Exercise"))

        # ── ROM Heatmap ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🌡️ Range of Motion Heatmap")
        chosen_ex_heat = st.selectbox(
            "Exercise for heatmap", df["exercise"].unique().tolist(), key="heat_ex")
        heat_df = df[df["exercise"] == chosen_ex_heat].tail(
            20).reset_index(drop=True)
        heat_df["rep_idx"] = heat_df.index + 1

        if not heat_df.empty and "depth_score" in heat_df.columns:
            melt_cols = ["rep_idx", "depth_score",
                         "alignment_score", "stability_score"]
            heat_melt = heat_df[melt_cols].melt(
                id_vars="rep_idx", var_name="Component", value_name="Score")
            heat_melt["Component"] = heat_melt["Component"].str.replace(
                "_score", "").str.capitalize()

            heatmap_chart = alt.Chart(heat_melt).mark_rect().encode(
                x=alt.X("rep_idx:O", title="Rep number"),
                y=alt.Y("Component:N", title=""),
                color=alt.Color("Score:Q",
                                scale=alt.Scale(
                                    scheme="viridis", domain=[0, 40]),
                                title="Score"),
                tooltip=["rep_idx", "Component", "Score"],
            ).properties(height=140)
            st.altair_chart(heatmap_chart, use_container_width=True)

        # ── Weekly Plan ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📅 Weekly Workout Plan")
        plan = generate_weekly_plan(profile, df)
        ex_colors = {"Squat": "#1a3a5c",
                     "Push-up": "#1a3a2a", "Shoulder Press": "#3a2a1a"}
        plan_cols = st.columns(5)
        for col, day in zip(plan_cols, plan):
            with col:
                st.markdown(f"**{day['day']}**")
                if day["rest"]:
                    st.caption("Rest 😴")
                else:
                    for ex in day["exercises"]:
                        bg = ex_colors.get(ex["name"], "#1a1e2a")
                        st.markdown(
                            f'<div style="background:{bg};border-radius:8px;padding:6px 10px;'
                            f'margin:4px 0;font-size:.82rem;">'
                            f'<strong>{ex["name"]}</strong><br>{ex["reps"]} reps'
                            f'{"<br><small style=color:#ffa;>" + ex["note"] + "</small>" if ex["note"] else ""}'
                            f'</div>',
                            unsafe_allow_html=True)

        # ── Deep dive ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Exercise Deep-Dive")
        chosen = st.selectbox(
            "Select exercise", df["exercise"].unique().tolist(), key="drill")
        ex_df = df[df["exercise"] == chosen]
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Reps", len(ex_df))
        d2.metric("Avg Score",  f"{ex_df['form_score'].mean():.1f}%")
        d3.metric("Improvement",
                  f"+{ex_df['form_score'].iloc[-1]-ex_df['form_score'].iloc[0]:.1f}pts"
                  if len(ex_df) > 1 else "—")
        st.area_chart(ex_df[["depth_score", "alignment_score",
                      "stability_score"]].reset_index(drop=True))

        # Tempo chart if data available
        if "eccentric_s" in ex_df.columns and ex_df["eccentric_s"].sum() > 0:
            st.markdown("##### Tempo (eccentric duration per rep)")
            st.line_chart(
                ex_df[["eccentric_s", "concentric_s"]].reset_index(drop=True))

        # ── Rep Clips ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎬 Rep Clips")
        clips = RepVideoExporter.list_clips(CLIPS_DIR)
        if not clips:
            st.caption("No clips yet — enable recording during a workout.")
        else:
            clip_cols = st.columns(min(3, len(clips)))
            for i, (col, clip) in enumerate(zip(cycle(clip_cols), clips[:9])):
                with col:
                    st.caption(os.path.basename(clip))
                    st.video(clip)

        # ── Raw data ──────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📥 Raw Session Data"):
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(),
                               "vcoach_sessions.csv", "text/csv")


# itertools.cycle needed for rep clips grid
from itertools import cycle  # noqa: E402
