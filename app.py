from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Directory to save uploaded images
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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/embedding", methods=["POST"])
def get_embedding():
    """
    Accepts JSON:
    {
        "image": "data:image/jpeg;base64,..."
    }
    Returns 128-d face embedding
    """
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    
    img_b64 = data["image"]
    img_path = save_base64_image(img_b64, "temp.jpg")
    
    try:
        embedding = DeepFace.represent(img_path, enforce_detection=True)[0]["embedding"]
        return jsonify({"embedding": embedding})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/faceembedding', methods=['POST'])
def get_face_embedding():
    try:
        data = request.json
        image_base64 = data.get("image")

        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        img_b64 = data["image"]
        img_path = save_base64_image(img_b64, "temp.jpg")
        
        # 🔍 Detect faces
        faces = DeepFace.extract_faces(img_path=img_path, enforce_detection=False)
        face_count = len(faces)
        print("Error1:")

        # 🧠 Get embeddings for all faces
        representations = DeepFace.represent(img_path=img_path, enforce_detection=False)

        embeddings = [rep["embedding"] for rep in representations]

        return jsonify({
            "face_count": face_count,
            "embeddings": embeddings
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/verify", methods=["POST"])
def verify_faces():
    """
    Accepts JSON:
    {
        "image1": "data:image/jpeg;base64,...",
        "image2": "data:image/jpeg;base64,..."
    }
    Returns verification result
    """
    data = request.json
    if "image1" not in data or "image2" not in data:
        return jsonify({"error": "Both images required"}), 400
    
    img1_path = save_base64_image(data["image1"], "img1.jpg")
    img2_path = save_base64_image(data["image2"], "img2.jpg")
    
    try:
        result = DeepFace.verify(img1_path, img2_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/match', methods=['POST'])
def match_faces():
    try:
        data = request.json
        image_base64 = data.get("image")
        patient_embeddings = data.get("patients")  # list of {id, embedding}

        if not image_base64 or not patient_embeddings:
            return jsonify({"error": "Missing image or patient embeddings"}), 400

        img_b64 = data["image"]
        img_path = save_base64_image(img_b64, "temp.jpg")

        # 🧠 Get embeddings for faces in image
        representations = DeepFace.represent(img_path=img_path, enforce_detection=False)

        results = []

        for i, face in enumerate(representations):
            face_embedding = face["embedding"]

            best_match = None
            best_score = -1

            for patient in patient_embeddings:
                score = cosine_similarity(face_embedding, patient["embedding"])

                if score > best_score:
                    best_score = score
                    best_match = patient["id"]

            results.append({
                "face_index": i,
                "matched_patient_id": best_match if best_score > 0.7 else None,
                "similarity": float(best_score)
            })

        return jsonify({
            "face_count": len(representations),
            "matches": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)