# BioAuth-Deepface

## Overview

A Python Flask backend that generates **face embeddings** and performs **face matching** using DeepFace (ArcFace model) and TensorFlow.

### Features

- Generate face embeddings from base64-encoded images
- Match faces against a list of registered users
- Multi-face detection support using MTCNN backend
- Cosine similarity-based matching with configurable threshold

---

## Project Structure

```
BioAuth-Deepface/
├── app.py              # Main Flask backend
├── requirements.txt    # Python dependencies
├── uploads/            # Temp storage for uploaded images
└── README.md
```

---

## Prerequisites

- Python 3.10
- pip >= 23
- Internet connection (for DeepFace pre-trained model download on first run)
- Optional: NVIDIA GPU + CUDA for faster inference

---

## Setup & Run

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the server
python app.py
```

The server starts at **http://localhost:5001**.

---

## API Endpoints

### `POST /getfaceembedding`

Returns the face embedding for a single-face image.

**Request:**

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**

```json
{
  "embedding": [0.12, -0.34, ...]
}
```

---

### `POST /facematch`

Detects all faces in an image and matches each against registered users.

**Request:**

```json
{
  "image": "data:image/jpeg;base64,...",
  "users": [
    { "id": "u1", "name": "John", "embedding": [0.12, -0.34, "..."] },
    { "id": "u2", "name": "Alice", "embedding": [0.56, 0.78, "..."] }
  ]
}
```

**Response:**

```json
{
  "face_count": 1,
  "matches": [
    {
      "face_index": 0,
      "name": "John",
      "user_id": "u1",
      "similarity": 0.87
    }
  ]
}
```

---

## Configuration

| Constant              | Default    | Description                        |
|-----------------------|------------|------------------------------------|
| `MODEL_NAME`          | `ArcFace`  | DeepFace embedding model           |
| `SIMILARITY_THRESHOLD`| `0.32`     | Minimum cosine similarity to match |
