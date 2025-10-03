# Facial Emotion Detection 🎭

This project detects human faces and recognizes their emotions in real-time using **OpenCV** and a pre-trained **CNN model**.

## Features
- Real-time webcam feed with emotion detection
- Tracks faces and applies smoothing for stability
- Supports multiple emotions (Happy, Sad, Angry, Surprise, Neutral, etc.)
- Uses Haar Cascade for face detection

## Requirements
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt

python obj.py

##Folder Structure
.
├── emotion_detection.py   # Main code
├── haarcascade_frontalface_default.xml
├── emotion_model.hdf5     # Pre-trained emotion model
├── requirements.txt
└── README.md
