from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import numpy as np
import os
import json
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_NAME = "ArcFace"
SIMILARITY_THRESHOLD = 0.32

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def get_patients_with_embeddings():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT p.uuid, p.uhid, pfe.embedding FROM dev_hmis_patients_18_12_2019.patient_face_embedding pfe JOIN patients p ON pfe.patient_uuid = p.uuid where pfe.is_active = 1"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    patients = []
    for row in rows:
        embedding = row["embedding"]
        while isinstance(embedding, (str, bytes)):
            embedding = json.loads(embedding)
        patients.append({
            "uuid": row["uuid"],
            "embedding": embedding,
        })
    return patients


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


@app.route('/api/getfaceembedding', methods=['POST'])
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


@app.route('/api/facematch', methods=['POST'])
def face_match():
    """
    Accepts JSON:
    {
        "image": "data:image/jpeg;base64,..."
    }
    Detects all faces in the image and matches each face to the closest
    patient from the database. Returns the name for each detected face.
    """
    try:
        data = request.json
        if not data or not data.get("image"):
            return jsonify({"error": "Missing image"}), 400

        img_path = save_base64_image(data["image"], "temp.jpg")

        representations = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend='mtcnn'
        )

        if len(representations) == 0:
            return jsonify({"error": "No faces detected"}), 400

        patients = get_patients_with_embeddings()
        if not patients:
            return jsonify({"error": "No patients with embeddings found in database"}), 404

        threshold = SIMILARITY_THRESHOLD
        results = []

        for i, face in enumerate(representations):
            face_embedding = face["embedding"]

            best_uuid = None
            best_score = -1

            for patient in patients:
                score = cosine_similarity(face_embedding, patient["embedding"])
                if score > best_score:
                    best_score = score
                    best_uuid = patient["uuid"]

            if best_score > threshold:
                results.append({
                    "face_index": i,
                    "patient_uuid": best_uuid,
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
