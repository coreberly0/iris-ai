from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# ---------------- DECODE ----------------
def decode_image(base64_str):
    try:
        base64_str = base64_str.split(",")[-1]
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        return img
    except:
        return None


# ---------------- NORMALIZE ----------------
def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# ---------------- FEATURE EXTRACTION (IMPROVED) ----------------
def extract_features(img):
    img = cv2.resize(img, (128, 128))

    # CLAHE (better contrast than equalizeHist)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # center crop
    h, w = img.shape
    cx, cy = w // 2, h // 2
    crop = img[cy-50:cy+50, cx-50:cx+50]

    if crop.shape != (100, 100):
        crop = cv2.resize(crop, (100, 100))

    # edges
    edges = cv2.Canny(crop, 60, 120)

    # blur reduce noise
    crop = cv2.GaussianBlur(crop, (3, 3), 0)

    # flatten + normalize
    vec1 = crop.flatten() / 255.0
    vec2 = edges.flatten() / 255.0

    combined = np.concatenate([vec1, vec2])

    return normalize(combined).tolist()


@app.route("/extract", methods=["POST"])
def extract():
    data = request.json
    iris = data.get("irisImage")

    if not iris:
        return jsonify({"error": "No image"}), 400

    img = decode_image(iris)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    return jsonify({
        "iris_vector": extract_features(img)
    })


@app.route("/")
def home():
    return "Iris AI Running ✔"


if __name__ == "__main__":
    app.run(debug=True)