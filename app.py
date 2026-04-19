"""
Iris AI Service  —  app.py
===========================
No eye side. No left/right. Just extract features from whatever iris is presented.

ENDPOINTS
─────────
GET  /          → health check
POST /extract   → body: { irisImage: "<base64 BMP>" }
                  returns: { iris_vector, vector_length }
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import traceback
import os

app = Flask(__name__)
CORS(app)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Iris AI Running"})


# ── Extract ───────────────────────────────────────────────────────────────────
@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.get_json()

        if not data or "irisImage" not in data:
            return jsonify({"error": "Missing irisImage"}), 400

        img = b64_to_img(data["irisImage"])
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        iris   = crop_iris(img)
        vector = make_vector(iris)

        print(f"[EXTRACT] dims={len(vector)}")

        return jsonify({
            "success":       True,
            "iris_vector":   vector,
            "vector_length": len(vector),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Base64 → OpenCV ───────────────────────────────────────────────────────────
def b64_to_img(b64):
    try:
        if "," in b64:
            b64 = b64.split(",")[1]
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[DECODE ERROR] {e}")
        return None


# ── Detect and crop iris to 64×64 gray ───────────────────────────────────────
def crop_iris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1, minDist=50,
        param1=50, param2=28,
        minRadius=int(min(h, w) * 0.15),
        maxRadius=int(min(h, w) * 0.45),
    )

    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype("int")
        pad = int(r * 0.1)
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - pad)
        x2 = min(w, cx + r + pad)
        y2 = min(h, cy + r + pad)
        crop = gray[y1:y2, x1:x2]
        print(f"[CROP] circle ({cx},{cy}) r={r}")
    else:
        print("[CROP] no circle detected — using centre region")
        mid_y, mid_x = h // 2, w // 2
        r    = int(min(h, w) * 0.35)
        crop = gray[mid_y - r: mid_y + r, mid_x - r: mid_x + r]

    return cv2.resize(crop, (64, 64))


# ── Build feature vector ──────────────────────────────────────────────────────
def make_vector(iris):
    """
    Returns a 264-element L2-normalised feature vector.

    Dimensions
    ──────────
    LBP full image       64   micro-texture
    Gabor bank (6a×2s)   24   directional texture
    LBP left-half        32   asymmetric texture left side
    LBP right-half       32   asymmetric texture right side
    Radial zones (3×16)  48   inner / mid / outer ring
    DCT 8×8              64   frequency fingerprint
    ─────────────────────────
    Total               264   (L2 normalised)
    """

    # Contrast normalise
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq    = clahe.apply(iris)
    norm  = eq.astype(np.float32) / 255.0

    # 1. LBP full (64 bins)
    lbp_full = compute_lbp(eq)
    h64      = norm_hist(lbp_full.ravel(), 64, (0, 256))

    # 2. Gabor bank (6 angles × 2 scales × 2 stats = 24 values)
    gabor_feats = []
    for deg in [0, 30, 60, 90, 120, 150]:
        theta = np.deg2rad(deg)
        for lambd in [8.0, 16.0]:
            k = cv2.getGaborKernel(
                (21, 21), sigma=4.5,
                theta=theta, lambd=lambd,
                gamma=0.5, psi=0
            )
            resp = cv2.filter2D(norm, cv2.CV_32F, k)
            gabor_feats += [float(resp.mean()), float(resp.std())]
    gabor_feats = np.array(gabor_feats, np.float32)

    # 3. Half-image LBP (32 bins each)
    left_lbp  = compute_lbp(eq[:, :32])
    right_lbp = compute_lbp(eq[:, 32:])
    h_left    = norm_hist(left_lbp.ravel(),  32, (0, 256))
    h_right   = norm_hist(right_lbp.ravel(), 32, (0, 256))

    # 4. Radial zones (3 rings × 16 bins = 48 values)
    hy, hx = eq.shape[0] // 2, eq.shape[1] // 2
    max_r  = min(hx, hy)
    prev   = np.zeros_like(eq)
    zones  = []
    for frac in [0.33, 0.66, 1.0]:
        mask = np.zeros_like(eq)
        cv2.circle(mask, (hx, hy), int(max_r * frac), 255, -1)
        ring = cv2.bitwise_and(mask, cv2.bitwise_not(prev))
        pix  = eq[ring > 0]
        zones.append(
            norm_hist(pix, 16, (0, 256)) if len(pix) > 0
            else np.zeros(16, np.float32)
        )
        prev = mask
    radial = np.concatenate(zones)

    # 5. DCT 8×8 (64 values)
    dct_block = cv2.dct(norm)[:8, :8].flatten().astype(np.float32)

    # Combine and L2-normalise
    combined = np.concatenate([h64, gabor_feats, h_left, h_right, radial, dct_block])
    n = np.linalg.norm(combined)
    if n > 1e-7:
        combined /= n

    return combined.tolist()


# ── Helpers ───────────────────────────────────────────────────────────────────
def norm_hist(pixels, bins, range_):
    h, _ = np.histogram(pixels, bins=bins, range=range_)
    h    = h.astype(np.float32)
    h   /= (h.sum() + 1e-7)
    return h


def compute_lbp(img):
    h, w  = img.shape
    out   = np.zeros((h, w), np.uint8)
    nbrs  = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            c, code = img[i, j], 0
            for bit, (di, dj) in enumerate(nbrs):
                if img[i + di, j + dj] >= c:
                    code |= (1 << bit)
            out[i, j] = code
    return out


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print(f"=== Iris AI Service running on port {port} ===")
    app.run(host="0.0.0.0", port=port, debug=False)