import os
from pathlib import Path

project_root = "Facial_Emotion_Recognition"

folders = [
    "data",
    "models",
    "real_time"
]

files = [
    "train_model.ipynb",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "real_time/webcam_emotion.py"
]

# Create folders
for folder in folders:
    path = Path(project_root) / folder
    path.mkdir(parents=True, exist_ok=True)

# Create empty files
for file in files:
    path = Path(project_root) / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

print("✅ Project structure created successfully!")