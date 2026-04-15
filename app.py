from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# ---------------- DECODE BASE64 ----------------
def decode_image(base64_str):
    try:
        base64_str = base64_str.split(",")[-1]

        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        return img
    except Exception as e:
        print("DECODE ERROR:", e)
        return None


# ---------------- FEATURE EXTRACTION (STABLE VERSION) ----------------
def extract_features(img):

    # 1. resize fixed
    img = cv2.resize(img, (128, 128))

    # 2. normalize lighting
    img = cv2.equalizeHist(img)

    # 3. center crop (more stable size)
    h, w = img.shape
    cx, cy = w // 2, h // 2

    crop = img[cy-50:cy+50, cx-50:cx+50]

    if crop.shape != (100, 100):
        crop = cv2.resize(crop, (100, 100))

    # 4. blur noise
    crop = cv2.GaussianBlur(crop, (5, 5), 0)

    # 5. edges
    edges = cv2.Canny(crop, 40, 120)

    # 6. normalize
    crop = crop.astype(np.float32) / 255.0
    edges = edges.astype(np.float32) / 255.0

    # 7. flatten
    crop_vec = crop.flatten()
    edge_vec = edges.flatten()

    # 8. combine
    vector = np.concatenate((crop_vec, edge_vec))

    # 9. IMPORTANT: L2 NORMALIZATION (FIXES YOUR BUG)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector.tolist()


# ---------------- API ----------------
@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.get_json()
        iris = data.get("irisImage")

        if not iris:
            return jsonify({"error": "No image"}), 400

        img = decode_image(iris)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        vector = extract_features(img)

        return jsonify({
            "iris_vector": vector,
            "length": len(vector)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ---------------- HEALTH ----------------
@app.route("/", methods=["GET"])
def home():
    return "Iris AI Running ✅"


if __name__ == "__main__":
     app.run(debug=True)