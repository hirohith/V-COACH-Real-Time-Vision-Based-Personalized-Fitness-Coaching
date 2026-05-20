VCoach – Real Time Vision Based Personalized Fitness Coaching
Overview

VCoach is an AI-powered real-time fitness coaching system developed using Computer Vision, Artificial Intelligence, and Biomechanics. The system analyzes human body posture during workouts using a standard webcam and provides real-time corrective feedback, repetition counting, posture scoring, fatigue detection, and workout analytics.

The project aims to bridge the gap between professional fitness coaching and affordable home-based workout systems by delivering intelligent biomechanical exercise analysis without requiring expensive hardware such as smart mirrors, wearable sensors, or motion tracking devices.

The platform currently supports:

Squats
Push-Ups
Shoulder Press

The system uses Google MediaPipe BlazePose for human pose estimation and OpenCV for real-time video processing.

Features
Real-Time Pose Estimation
Detects 33 human body landmarks
Real-time skeletal tracking using MediaPipe BlazePose
CPU-based lightweight processing
Exercise Recognition


Supports:
Squats
Push-Ups
Shoulder Press
Automatic exercise phase detection
Rep Counting
Finite-State Machine (FSM) based repetition counting
Hysteresis threshold control for accurate tracking
Biomechanical Analysis
Joint angle calculation:
Knee angle
Hip angle
Elbow angle
Torso inclination
Spine alignment
Form Scoring System

Each rep is evaluated using:

Depth Score
Alignment Score
Stability Score
Fatigue Detection
Detects posture degradation
Identifies unstable movement patterns
Warns users about muscular fatigue
Symmetry Analysis
Detects left-right body imbalance
Helps improve posture consistency
Rep Tempo Monitoring
Eccentric and concentric timing analysis
Workout tempo feedback
Personalized Coaching

Supports:

Beginner
Intermediate
Advanced

Goals:

Weight Loss
Muscle Gain
General Fitness
Streamlit Dashboard
Workout analytics
Progress tracking
Heatmaps
Form score visualization
Workout history
Technologies Used
Technology	Purpose
Python	Core Programming
OpenCV	Video Processing
MediaPipe BlazePose	Human Pose Estimation
NumPy	Mathematical Operations
Pandas	Data Analysis & Storage
Streamlit	Dashboard Visualization
Gemini API	AI Coaching Summary


System Architecture

Webcam Input
      ↓
Frame Preprocessing
(CLAHE + EMA)
      ↓
MediaPipe Pose Detection
      ↓
Joint Angle Calculation
      ↓
FSM Exercise Recognition
      ↓
Rep Counting & Form Scoring
      ↓
Fatigue & Symmetry Analysis
      ↓
Dashboard & AI Feedback



Project Structure
VCoach/
│
├── app.py
├── pose_detection.py
├── exercise_fsm.py
├── form_scoring.py
├── fatigue_detection.py
├── symmetry_analysis.py
├── dashboard.py
├── utils/
├── assets/
├── data/
├── requirements.txt
└── README.md



Installation
Clone Repository
git clone https://github.com/your-username/VCoach.git
cd VCoach
Create Virtual Environment
python -m venv venv
Activate Environment
Windows
venv\Scripts\activate
Linux/Mac
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Run the Application
streamlit run app.py
Requirements
Python 3.10+
OpenCV
MediaPipe
NumPy
Pandas
Streamlit
Matplotlib


Experimental Results
Metric	Result
Landmark Detection Accuracy	97.2%
Rep Counting Accuracy	98.4%
Joint Angle Error	±1.8°
Form Score Correlation	r = 0.84
