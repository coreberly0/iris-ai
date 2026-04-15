from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# ---------------- DECODE BASE64 ----------------
def decode_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    return img

# ---------------- EXTRACT IRIS FEATURES ----------------
def extract_features(img):
    # ✅ Resize (important for consistency)
    img = cv2.resize(img, (64, 64))

    # ✅ Improve contrast
    img = cv2.equalizeHist(img)

    # ✅ Reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # ✅ Edge detection (very important)
    edges = cv2.Canny(img, 50, 150)

    # ✅ Combine original + edges
    combined = np.concatenate((img.flatten(), edges.flatten()))

    return combined.tolist()

# ---------------- API ----------------
@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.json
        iris = data["irisImage"]

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