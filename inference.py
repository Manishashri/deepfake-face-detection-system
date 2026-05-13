import matplotlib
matplotlib.use("Agg")

import os, json
import numpy as np
import torch
import cv2
import joblib
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import torch.nn as nn
import torchvision.models as models

# 🔥 IMPORTANT (FIX PICKLE ERROR)
from model.lda import MultiHeadLDA


# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# ENCODER
# -----------------------
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        f = self.backbone(x)
        return f.view(B, T, -1)


encoder = ResNetEncoder().to(device)
encoder.load_state_dict(torch.load("model/encoder.pth", map_location=device))
encoder.eval()


# -----------------------
# LOAD MODELS
# -----------------------
lda = joblib.load("model/mh_lda.pkl")   # ORIGINAL MODEL
clf = joblib.load("model/logreg.pkl")


# -----------------------
# FACE DETECTOR
# -----------------------
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)


# -----------------------
# OUTPUT DIR
# -----------------------
RESULT_DIR = "static"
os.makedirs(RESULT_DIR, exist_ok=True)


# -----------------------
# FRAME LEVEL PREDICTION
# -----------------------
def frame_level_predictions(video_path, frames_per_video=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 🔥 SAME AS COLAB (VERY IMPORTANT)
    idxs = np.linspace(0, max(total-1, 0), frames_per_video).astype(int)

    probs = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        face = mtcnn(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if face is None:
            continue

        face = face.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = encoder(face).mean(dim=1).cpu().numpy()

        prob = clf.predict_proba(lda.transform(feat))[0][1]
        probs.append(float(prob))

    cap.release()
    return probs


# -----------------------
# SAVE GRAPH
# -----------------------
def save_frame_plot(probs):
    plt.figure(figsize=(6,4))
    plt.plot(probs, marker="o")
    plt.axhline(0.5, linestyle="--", color="red")
    plt.xlabel("Frame Index")
    plt.ylabel("Fake Probability")
    plt.title("Frame-wise Fake Probability")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/frame_probs.png")
    plt.close()


# -----------------------
# MAIN INFERENCE
# -----------------------
def run_full_inference(video_path):
    probs = frame_level_predictions(video_path)

    if len(probs) == 0:
        result = {
            "video": os.path.basename(video_path),
            "label": "NO FACE",
            "confidence": 0.0,
            "num_frames_used": 0
        }
    else:
        final_prob = float(np.mean(probs))

        # 🔥 Slight improvement for real-world videos
        if final_prob > 0.6:
            label = "FAKE"
        else:
            label = "REAL"

        save_frame_plot(probs)

        result = {
            "video": os.path.basename(video_path),
            "label": label,
            "confidence": round(final_prob, 3),
            "num_frames_used": len(probs)
        }

    with open(f"{RESULT_DIR}/prediction.json", "w") as f:
        json.dump(result, f, indent=4)

    return result