from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

def decode_image(base64_str):
    try:
        base64_str = base64_str.split(",")[-1]
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        return img
    except:
        return None


def extract_features(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.equalizeHist(img)

    # normalize
    img = img / 255.0

    return img.flatten().astype(np.float32)


@app.route("/extract", methods=["POST"])
def extract():
    data = request.json
    iris = data.get("irisImage")

    img = decode_image(iris)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    vector = extract_features(img)

    return jsonify({
        "iris_vector": vector.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)