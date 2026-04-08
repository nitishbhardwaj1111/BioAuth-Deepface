# Face Detection & Embedding Project (Python Backend Only)

## Overview

This project captures a user's face image and generates a **face embedding** using DeepFace and TensorFlow.  
It is built with Python and Flask.

### Features

- Upload an image via REST API
- Generate face embedding
- Detect if a face cannot be detected
- Optional: enforce_detection can be set to False

---

## Project Structure
face-detect-python/
│
├─ app.py # Main Flask backend file
├─ requirements.txt # Python dependencies
├─ uploads/ # Folder to temporarily store uploaded images
└─ README.md # This file


---

## Prerequisites

- Python 3.10
- pip >= 23
- CPU with AVX/AVX2 (TensorFlow CPU optimized)
- Optional: NVIDIA GPU + CUDA (for faster inference)
- Internet connection (for DeepFace pre-trained models download)

---

## Setup Instructions

### 1. Clone / Copy Project

Copy the `face-detect-python` folder to your system or server.

### 2. Create Virtual Environment
## Quick Setup Summary
# Step 1: Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Step 2: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Run backend
python app.py