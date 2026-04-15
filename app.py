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

# ---------------- EXTRACT FEATURES ----------------
def extract_features(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 50, 150)

    combined = np.concatenate((img.flatten(), edges.flatten()))
    return combined.tolist()

# ---------------- API ----------------
@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.json
        iris = data.get("irisImage")

        if not iris:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image(iris)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        vector = extract_features(img)

        return jsonify({"iris_vector": vector})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Iris AI Running ✅"

if __name__ == "__main__":
    app.run(debug=True)