from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_NAME = "ArcFace"
SIMILARITY_THRESHOLD = 0.32

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_base64_image(base64_str, filename):
    img_data = base64.b64decode(base64_str.split(',')[1])
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(img_data)
    return filepath


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@app.route('/getfaceembedding', methods=['POST'])
def get_face_embedding():
    """
    Accepts JSON:
    {
        "image": "data:image/jpeg;base64,..."
    }
    Returns the face embedding for a single-face image.
    """
    try:
        data = request.json
        if not data or not data.get("image"):
            return jsonify({"error": "No image provided"}), 400

        img_path = save_base64_image(data["image"], "temp.jpg")

        representations = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend='mtcnn'
        )

        if len(representations) == 0:
            return jsonify({"error": "No face detected"}), 400

        if len(representations) > 1:
            return jsonify({"error": "Multiple faces detected, expected single face"}), 400

        return jsonify({
            "embedding": representations[0]["embedding"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/facematch', methods=['POST'])
def face_match():
    """
    Accepts JSON:
    {
        "image": "data:image/jpeg;base64,...",
        "users": [
            {"id": "u1", "name": "John", "embedding": [0.12, -0.34, ...]},
            {"id": "u2", "name": "Alice", "embedding": [0.56, 0.78, ...]}
        ]
    }
    Detects all faces in the image and matches each face to the closest
    registered user. Returns the name for each detected face.
    """
    try:
        data = request.json
        if not data or not data.get("image") or not data.get("users"):
            return jsonify({"error": "Missing image or users list"}), 400

        img_path = save_base64_image(data["image"], "temp.jpg")

        representations = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend='mtcnn'
        )

        if len(representations) == 0:
            return jsonify({"error": "No faces detected"}), 400

        threshold = SIMILARITY_THRESHOLD
        users = data["users"]
        results = []

        for i, face in enumerate(representations):
            face_embedding = face["embedding"]

            best_id = None
            best_name = None
            best_score = -1

            for user in users:
                score = cosine_similarity(face_embedding, user["embedding"])
                if score > best_score:
                    best_score = score
                    best_id = user["id"]
                    best_name = user["name"]

            if best_score > threshold:
                results.append({
                    "face_index": i,
                    "name": best_name,
                    "user_id": best_id,
                    "similarity": best_score
                })

        return jsonify({
            "face_count": len(representations),
            "matches": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
