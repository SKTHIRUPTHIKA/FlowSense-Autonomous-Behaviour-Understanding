# FlowSense-Autonomous-Behaviour-Understanding

**Computer Vision · Action Recognition · Object Tracking · Behaviour Analysis**

## Overview

FlowSense is an autonomous CCTV vision and behaviour-understanding system developed for:

**HNX26PSI07 – Autonomous Vision & Behaviour Understanding**

The system processes CCTV video to detect people, track them with persistent IDs, analyze their behaviour over time, and generate explainable abnormal-behaviour and critical-risk alerts.

### Core Pipeline

```text
CCTV Video
    ↓
Person Detection
    ↓
Object Tracking
    ↓
Persistent Person IDs
    ↓
Temporal Behaviour Analysis
    ↓
Activity Recognition
    ↓
Normal / Abnormal Analysis
    ↓
Risk Scoring

Key Features
1. Person Detection
FlowSense detects people in CCTV footage using a YOLO-based object detection model.
The current demonstration focuses on the person class.
2. Multi-Person Tracking
Detected people are tracked across video frames using:
- BoT-SORT
- ReID-based appearance matching
Each tracked person receives a persistent tracking ID, for example:
ID 1
ID 4
ID 15
ID 28

ReID is used for tracking consistency and is not facial recognition.
3. Temporal Behaviour Analysis
FlowSense analyzes a person's position and movement over multiple frames instead of making a decision from a single frame.
The current MVP focuses on prolonged presence within a local spatial area.
When a person remains within the configured area for longer than the threshold, the system identifies:
LOITERING

Current configuration:
Local area: approximately 5-metre equivalent
Loitering threshold: 10 seconds

The 5-metre value is an image-based approximation and is not a calibrated physical distance.
4. Activity Recognition
A pretrained YOLO26s-Pose model is used for basic pose-based activity analysis.
Current activity states:
WALKING
STANDING

5. Explainable Risk Score
The system uses a deterministic rule-based risk score.
For detected loitering:
Initial risk = 60
Risk increases with continued loitering
Maximum risk = 100

The current critical threshold is:
Risk > 80

When the threshold is crossed, FlowSense generates a critical-risk signal.
The risk score represents a rule-based assessment, not a probability of danger.
6. Critical Alerts
When a person's risk exceeds the configured threshold, the dashboard displays:
CRITICAL ALERT

The alert includes:
- Person ID
- Behaviour
- Activity
- Duration
- Risk score
- Timestamp
Technologies Used
- Python 3.11
- OpenCV
- NumPy
- Pandas
- PyTorch
- Torchvision
- Ultralytics
- YOLO26s
- YOLO26s-Pose
- BoT-SORT
- ReID
- Computer Vision
- Object Tracking
- Temporal Behaviour Analysis
Models
YOLO26s
A pretrained YOLO26s model is used for person detection and tracking in the final demonstration.
YOLO26s-Pose
A pretrained YOLO26s-Pose model is used for pose-based activity analysis.
The current implementation uses human keypoints such as knees and ankles to estimate basic movement activity.
BoT-SORT + ReID
BoT-SORT maintains object tracks across frames.
ReID provides appearance-based information to improve track association.
Configuration:
05_configs/botsort_reid.yaml

Dataset
The project uses publicly available pedestrian/CCTV datasets, including:
- MOT17
- MOT20
- CrowdHuman
The prepared dataset contains:
39,296 image-label pairs

Dataset split:
Training:    27,507
Validation:   7,859
Testing:      3,930

The complete dataset is not included in this Git repository because of its large size.
Data Pipeline
Public Pedestrian / CCTV Data
          ↓
Raw Data Collection
          ↓
Dataset Preparation
          ↓
Person-Class Processing
          ↓
Annotation Preparation
          ↓
Train / Validation / Test Split
          ↓
Dataset Quality Check
          ↓
Model Configuration
          ↓
Detection & Tracking
          ↓
Behaviour Analysis

Dataset preparation scripts are available in:
06_scripts/

Important scripts include:
crowdhuman_prepare_person.py
mot17_prepare_person.py
mot20_prepare_person.py
fix_mot_labels.py
finalize_dataset.py

Project Structure
FlowSense/
│
├── 05_configs/
│   ├── botsort_reid.yaml
│   ├── data.yaml
│   └── data_small.yaml
│
├── 06_scripts/
│   ├── behavior_loitering.py
│   ├── crowdhuman_prepare_person.py
│   ├── final_demo.py
│   ├── final_demo_activity.py
│   ├── final_demo_critical.py
│   ├── final_demo_pose.py
│   ├── finalize_dataset.py
│   ├── fix_mot_labels.py
│   ├── mot17_prepare_person.py
│   └── mot20_prepare_person.py
│
├── README.md
├── requirements.txt
└── .gitignore

Large datasets, model weights, checkpoints, generated results and videos are excluded using .gitignore.
Installation
Requirements
- Python 3.11
- Git
- Windows/Linux
- Sufficient storage and RAM
Create Virtual Environment
python -m venv venv

Windows
venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt

Configuration
Configuration files are located in:
05_configs/

Dataset
05_configs/data.yaml

Small Dataset
05_configs/data_small.yaml

BoT-SORT + ReID
05_configs/botsort_reid.yaml

The demonstration scripts contain configuration values for:
- Input video
- Output video
- Loitering threshold
- Local-area radius
- Movement threshold
- Critical-risk threshold
Paths may need to be updated when running the project on another computer.
Running the System
Critical Alert Demo
Run:
python 06_scripts/final_demo_critical.py

The system processes the configured CCTV video and generates the FlowSense behaviour dashboard.
The output displays:
- Person detection
- Tracking IDs
- Activity
- Behaviour
- Risk score
- Abnormal behaviour alerts
- Critical alerts
- Timestamp
Pose-Based Activity Demo
Run:
python 06_scripts/final_demo_pose.py

This version uses:
YOLO26s-Pose
+
BoT-SORT + ReID
+
Pose Keypoint Analysis

to estimate basic activity states.
Behaviour Detection Logic
For every tracked person:
Person detected
       ↓
Tracking ID assigned
       ↓
Position monitored
       ↓
Local area established
       ↓
Time within area measured
       ↓
Duration >= threshold?
       ↓
      YES
       ↓
LOITERING
       ↓
Risk increases
       ↓
Risk > 80?
       ↓
CRITICAL ALERT

This provides a deterministic and explainable reasoning mechanism.
Evidence & Explainability
FlowSense provides evidence for detected events through:
- Person ID
- Behaviour
- Activity
- Duration
- Risk score
- Video timestamp
Example:
ID 15
Behaviour: LOITERING
Activity: STANDING
Duration: 15.4 seconds
Risk: 81/100
Status: CRITICAL

Therefore, an evaluator can determine:
WHO    → Person ID
WHAT   → Behaviour
WHEN   → Timestamp / Duration
RISK   → Risk score
WHY    → Behaviour exceeded the configured temporal threshold

Sample Input & Output
Input
A CCTV-style pedestrian video containing multiple people moving through a scene.
Output
FlowSense produces a processed video containing:
Person Bounding Boxes
Tracking IDs
Activity Labels
Behaviour Labels
Risk Scores
Abnormal Behaviour Alerts
Critical Alerts
Timestamps

Example:
ID 15 | LOITERING
Activity: STANDING
Duration: 15.4s
Risk: 81/100

Experimental Custom Training
A small custom YOLO26n training experiment was performed to validate the dataset-to-model pipeline.
Configuration:
Training images: 100
Validation images: 20
Epochs: 1
Device: CPU

The training pipeline completed successfully.
However, the validation performance was low because the experiment used a very small dataset and only one training epoch.
Therefore, the pretrained YOLO26s model was used for the final demonstration because it provided more reliable person detection and tracking.
This distinction is documented to avoid presenting the experimental model as a high-accuracy trained model.
Minimum Viable Solution
The implemented MVP demonstrates:
- CCTV video input
- Person detection
- Multi-person tracking
- Persistent tracking IDs
- Temporal behaviour reasoning
- Loitering detection
- Basic activity recognition
- Explainable risk scoring
- Critical alerts
- Dashboard visualization
The core MVP pipeline is:
Detection
   ↓
Tracking
   ↓
Persistent ID
   ↓
Temporal Behaviour Analysis
   ↓
Loitering Detection
   ↓
Risk Scoring
   ↓
Critical Alert

Scope Note
Implemented MVP
The current working implementation includes:
- Person detection
- BoT-SORT tracking
- ReID-based tracking assistance
- Persistent tracking IDs
- Temporal local-area analysis
- Loitering detection
- Walking/standing activity analysis
- Deterministic risk scoring
- Critical alerts
- Timestamped dashboard output
Future / Stretch Goals
The following are planned extensions and are not claimed as fully implemented in the current MVP:
- Fall detection
- Sitting detection
- Running detection
- Restricted-zone intrusion
- Wrong-direction movement
- Crowding detection
- Multiple behaviour types
- General anomaly detection
- Camera calibration
- True physical-distance measurement
- Advanced action recognition
Limitations
Ground Truth
The demonstration video does not contain verified ground-truth loitering annotations.
Therefore, current loitering events represent:
Rule-based potential abnormal behaviour
and should not be interpreted as verified real-world abnormal events.
Distance
The approximately 5-metre local area is derived from image-space bounding-box information.
The camera is not geometrically calibrated, so it is not a true physical 5-metre measurement.
Activity Recognition
The current activity recognition focuses on:
WALKING
STANDING

More complex actions require additional training and/or temporal action-recognition models.
Tracking
BoT-SORT + ReID improves track association but does not guarantee perfect identity persistence after every long disappearance or re-entry.
Privacy
FlowSense does not perform face recognition.
It uses:
- Person detection
- Tracking IDs
- Appearance-based ReID
The tracking ID is a computational identity and does not represent the person's real-world identity.
External Resources
The project uses:
Datasets
- MOT17
- MOT20
- CrowdHuman
- CCTV-style pedestrian video data
Open-Source Frameworks
- Ultralytics
- OpenCV
- PyTorch
Pretrained Models
- YOLO26s
- YOLO26s-Pose
Tracking
- BoT-SORT
- ReID-based appearance matching
- ByteTrack was evaluated during development
All significant external datasets, pretrained models and open-source components are declared here as required by the submission guidelines.
PS07 Alignment
PS07 Requirement	FlowSense Implementation
Detect people/objects	YOLO-based person detection
Track entities	BoT-SORT + ReID
Persistent identity	Tracking IDs
Understand behaviour over time	Temporal spatial analysis
Normal vs unusual	Behaviour state classification
Identify entity	Person tracking ID
Identify when	Video timestamp and duration
Explain abnormal behaviour	Behaviour + duration + risk
Risk assessment	Deterministic risk score
Alert generation	Critical-risk signal
Evidence	ID, timestamp, duration and risk


Final Architecture
                 CCTV VIDEO
                     │
                     ▼
             YOLO PERSON DETECTION
                     │
                     ▼
              BoT-SORT + ReID
                     │
                     ▼
             PERSISTENT TRACK IDs
                     │
                     ▼
          TEMPORAL POSITION ANALYSIS
                     │
                     ▼
             ACTIVITY RECOGNITION
                     │
                     ▼
          NORMAL / ABNORMAL ANALYSIS
                     │
                     ▼
             LOITERING DETECTION
                     │
                     ▼
             EXPLAINABLE RISK SCORE
                     │
                     ▼
              CRITICAL ALERT
                     │
                     ▼
             FLOWSENSE DASHBOARD

Conclusion
FlowSense demonstrates an end-to-end prototype for autonomous CCTV vision and behaviour understanding.
It combines:
Detection + Tracking + Persistent IDs + Temporal Reasoning + Activity Analysis + Behaviour Detection + Explainable Risk + Alerting
The current MVP focuses on prolonged local-area presence as a potential loitering behaviour and demonstrates how raw CCTV video can be transformed into an explainable behaviour alert.
