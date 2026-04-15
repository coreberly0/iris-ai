from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# ---------------- DECODE BASE64 ----------------
def decode_image(base64_str):
    try:
        # ✅ remove data:image prefix
        base64_str = base64_str.split(",")[-1]

        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        return img
    except Exception as e:
        print("DECODE ERROR:", e)
        return None


# ---------------- EXTRACT IRIS FEATURES ----------------
def extract_features(img):
    # ✅ resize bigger for better detail
    img = cv2.resize(img, (128, 128))

    # ✅ normalize brightness
    img = cv2.equalizeHist(img)

    # ✅ take center crop (focus iris area)
    h, w = img.shape
    cx, cy = w // 2, h // 2

    crop = img[cy-40:cy+40, cx-40:cx+40]

    # safety check
    if crop.shape != (80, 80):
        crop = cv2.resize(crop, (80, 80))

    # ✅ reduce noise
    crop = cv2.GaussianBlur(crop, (5, 5), 0)

    # ✅ edge detection
    edges = cv2.Canny(crop, 50, 150)

    # ✅ normalize values (important!)
    crop = crop / 255.0
    edges = edges / 255.0

    # ✅ combine features
    combined = np.concatenate((crop.flatten(), edges.flatten()))

    return combined.tolist()


# ---------------- API ----------------
@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.json
        iris = data.get("irisImage")

        if not iris:
            return jsonify({"error": "No image"}), 400

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