import numpy as np

class MultiHeadLDA:
    def __init__(self, heads=4):
        self.heads = heads
        self.models = []

    def transform(self, X):
        outputs = []
        for lda, s, e in self.models:
            outputs.append(lda.transform(X[:, s:e]))
        return np.concatenate(outputs, axis=1)
from flask import Flask, render_template, request, jsonify
import os, json

# 🔥 IMPORT YOUR INFERENCE FUNCTION
from inference import run_full_inference   # we create this below

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_DIR = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # 🔥 RUN MODEL
    result = run_full_inference(video_path)

    return jsonify(result)

@app.route("/result")
def get_result():
    with open(f"{RESULT_DIR}/prediction.json") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(debug=True)
