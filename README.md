# Deepfake Face Detection System

## Overview
This project presents a hybrid deepfake face detection framework developed using deep learning, discriminative learning, and transformer-based contextual analysis. The system detects whether an uploaded facial video is REAL or FAKE by analyzing spatial artifacts and temporal inconsistencies present across video frames.

The proposed framework combines:
- ResNet-50 for deep feature extraction
- Multi-Head Linear Discriminant Analysis (MH-LDA)
- Temporal Aggregation
- Logistic Regression
- CViT2 (Convolutional Vision Transformer Version 2)

The integration of CViT2 improved the overall detection accuracy from 88.75% to 95.63%.

---

# Features

- Upload and analyze video files
- Face detection using MTCNN
- Deep feature extraction using ResNet-50
- Multi-Head LDA based discriminative learning
- CViT2 transformer-based enhancement
- REAL / FAKE prediction with confidence score
- Frame-wise fake probability visualization
- ROC Curve and performance analysis dashboard
- Flask-based web interface

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-learn
- Flask
- Matplotlib
- FaceNet-PyTorch
- Google Colab

---

# Dataset Used

The system was trained and evaluated using:
- FaceForensics++
- Celeb-DF

---

# Project Workflow

1. Video Upload  
2. Frame Extraction  
3. Face Detection using MTCNN  
4. ResNet-50 Feature Extraction  
5. Multi-Head Feature Decomposition  
6. MH-LDA Transformation  
7. CViT2 Prediction  
8. Temporal Aggregation  
9. Logistic Regression Classification  
10. Final REAL / FAKE Prediction  

---

# Accuracy

| Model | Accuracy |
|------|------|
| Baseline MH-LDA Model | 88.75% |
| Proposed MH-LDA + CViT2 Model | 95.63% |

---

# Evaluation Metrics

The project was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

# Project Structure

```bash
deepfake-face-detection-system/
│
├── app.py
├── inference.py
├── fix_lda_pickle.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   ├── roc.png
│   ├── frame_probs.png
│   ├── bar_chart.png
│   ├── line_chart.png
│   └── screenshots/
│
├── model/
│   ├── encoder.pth
│   ├── mh_lda.pkl
│   └── logreg.pkl
│
└── README.md

