from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from anomaly_detector import AnomalyDetector

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "outputs" / "index.faiss"

GOOD_FOLDER = PROJECT_ROOT / "data" / "test" / "good"

DEFECTIVE_FOLDER = PROJECT_ROOT / "data" / "test" / "defective"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# ==========================================================
# LOAD MODEL
# ==========================================================

detector = AnomalyDetector(INDEX_PATH)

# ==========================================================
# LOAD IMAGE PATHS
# ==========================================================

good_images = []

defective_images = []

for ext in IMAGE_EXTENSIONS:

    good_images.extend(GOOD_FOLDER.glob(f"*{ext}"))

    defective_images.extend(DEFECTIVE_FOLDER.glob(f"*{ext}"))

good_images = sorted(good_images)

defective_images = sorted(defective_images)

print(f"Good Images       : {len(good_images)}")
print(f"Defective Images  : {len(defective_images)}")

# ==========================================================
# RUN INFERENCE
# ==========================================================

y_true = []
scores = []

# ==========================================================
# GOOD IMAGES
# ==========================================================

for image_path in good_images:

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    distances, indices, image_score = detector.predict(image)

    scores.append(image_score)
    y_true.append(0)

# ==========================================================
# DEFECTIVE IMAGES
# ==========================================================

for image_path in defective_images:

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    distances, indices, image_score = detector.predict(image)

    scores.append(image_score)
    y_true.append(1)

scores = np.array(scores)
y_true = np.array(y_true)

# ==========================================================
# AUROC
# ==========================================================

auroc = roc_auc_score(
    y_true,
    scores
)

print()
print("=" * 60)
print(f"AUROC : {auroc:.4f}")
print("=" * 60)

# ==========================================================
# AUTOMATIC THRESHOLD
# ==========================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    scores
)

best_index = np.argmax(tpr - fpr)

threshold = thresholds[best_index]

print(f"Threshold : {threshold:.2f}")

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = (scores >= threshold).astype(int)

# ==========================================================
# METRICS
# ==========================================================

print()
print("=" * 60)

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")

print()
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("=" * 60)

# ==========================================================
# ROC CURVE
# ==========================================================

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUROC = {auroc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

save_path = PROJECT_ROOT / "outputs" / "roc_curve.png"

plt.savefig(save_path)

plt.show()

print()
print("ROC Curve Saved")
print(save_path)

print()
print("=" * 60)
print(f"Recommended Threshold = {threshold:.2f}")
print("=" * 60)