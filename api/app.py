from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from src.pipeline import run_pipeline, run_pipeline_debug

app = Flask(__name__)


@app.get("/health")
def health() -> tuple[str, int]:
    return "ok", 200


@app.get("/")
def index():
    payload = {
        "service": "sudoku-vision-api",
        "endpoints": ["/health", "/predict", "/predict_debug"],
        "ocr_modes": ["tesseract"],
        "example_curl": 'curl -X POST "http://localhost:8080/predict" -F "file=@/path/to/image.png"',
    }
    return jsonify(payload)


def _load_uploaded_image():
    if "file" not in request.files:
        return None, (jsonify({"error": "missing form field 'file'"}), 400)

    file = request.files["file"]
    if file.filename == "":
        return None, (jsonify({"error": "empty filename"}), 400)

    image_bytes = file.read()
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil), None


@app.post("/predict")
def predict():
    image, err = _load_uploaded_image()
    if err is not None:
        return err

    try:
        result = run_pipeline(image)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result), 200


@app.post("/predict_debug")
def predict_debug():
    image, err = _load_uploaded_image()
    if err is not None:
        return err

    try:
        result, debug_image = run_pipeline_debug(image)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    image_bgr = debug_image[:, :, ::-1]
    _, encoded = cv2.imencode(".png", image_bgr)
    debug_image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    return jsonify({"result": result, "debug_image_base64": debug_image_base64}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
