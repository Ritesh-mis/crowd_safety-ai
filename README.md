🚨 Crowd Safety AI
Real-time Crowd Risk Detection System - Problem Statement 1 ⭐
Live Risk Analysis Overlay - Hackathon Ready!

🎯 Team Information
🏆 Team Lead: Ratnesh Pathak
MSc IT Final Year | Thakur College of Science & Commerce, Mumbai
📧 pathakratnesh03@gmail.com | 📱 +91 8793614505 | Portfolio

🚀 Team Members:

Name	Role
Raunak Gupta	ML Engineer (Anomaly Detection)
Ayush Jha	Computer Vision (Density + Motion)
Ritesh Mishra	Visualization & UI Overlays
🔬 What We Built
Crowd Safety AI processes live video streams to detect crowd density anomalies, abnormal motion patterns, and classify risk levels in real-time:

text
LOW (0.0-0.3) → Normal crowd movement
MEDIUM (0.3-0.7) → Elevated density/speed  
HIGH (0.7-1.0) → ⚠️ Emergency risk detected
Live Demo Output: Risk: LOW | Score: 0.24 overlay on video feed 
​

🛠️ Tech Stack
bash
Python 3.11.8 | OpenCV 4.12.0 | NumPy 2.2.6 | scikit-learn 1.8.0
Module	Purpose	Algorithm
video_loader.py	Live frame extraction	OpenCV VideoCapture
preprocessing.py	Noise reduction	Gaussian Blur
density_estimation.py	Grid heatmaps	8x12 pixel analysis
motion_analysis.py	Flow detection	Farneback Optical Flow
anomaly_detection.py	Outlier scoring	Statistical deviation
risk_classifier.py	Risk levels	Threshold-based
visualizer.py	Live overlays	OpenCV drawing
🚀 60-Second Setup
bash
# Clone project
git clone <your-repo>
cd crowd_safety_ai

# Install
pip install -r requirements.txt

# LIVE WEBCAM DEMO (no video file needed!)
python src/main.py
Output: Live risk analysis with Risk: LOW/MEDIUM/HIGH overlays 🎥

📊 Pipeline Architecture
text
📹 Live Webcam/Video → Preprocessing → Density Grid (8x12) 
  ↓
💨 Motion Analysis → Anomaly Detection → Risk Score (0.0-1.0)
  ↓
🖥️ Visual Overlays → "Risk: LOW | Score: 0.24"
Modular Design: 8 independent, testable modules ✅

⚙️ Configuration (config.py)
python
# Risk Thresholds - TUNE FOR YOUR ENVIRONMENT
DENSITY_LOW, DENSITY_HIGH = 0.3, 0.7
MOTION_HIGH = 5.0  # pixels/frame
ANOMALY_THRESHOLD = 0.8

# Video Settings
FRAME_SKIP = 3
TARGET_SIZE = (640, 480)
🎖️ Hackathon Advantages
✅ Live Processing - No pre-recorded videos

✅ Privacy Safe - Density-based (no faces detected)

✅ Judge Friendly - Clear Risk: LOW | Score: X.XX overlays

✅ Modular - Easy to explain each component

✅ Scalable - Works on any camera feed

✅ Production Ready - Proper error handling

🏆 Ratnesh Pathak - Team Lead
MSc IT Final Year | Thakur College of Science & Commerce
Awards: Best Academic Student 2024 | Best Library Member 2024
Experience: 4+ years Cultural Committee leadership, Farewell 2025 Core Head
Internships: Business Dev, Data Science, Web Development
Skills: Python, OpenCV, ML, Flask/Django, Data Analysis

📱 +91 8793614505 | ✉️ pathakratnesh03@gmail.com

🔮 Future Roadmap
text
[ ] Multi-camera support
[ ] Real-time alerts (SMS/Email)
[ ] Cloud deployment (AWS)
[ ] Mobile app dashboard
[ ] Historical analytics
📸 Live Demo Screenshot
​

"This is normal crowd" → Safety confirmed ✅

bash
# ONE COMMAND HACKATHON DEMO
python src/main.py
Built with ❤️ by Team Ratnesh Pathak
MSc IT, Thakur College | Problem Statement 1