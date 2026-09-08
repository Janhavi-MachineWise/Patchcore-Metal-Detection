from pathlib import Path
import sys
import cv2

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# ==========================================================
# IMPORTS
# ==========================================================

from config import (
    THRESHOLD,
    IMAGE_EXTENSIONS
)

from anomaly_detector import AnomalyDetector
from heatmap_generator import HeatmapGenerator
from report_generator import ReportGenerator

# ==========================================================
# CONFIGURATION
# ==========================================================

INDEX_PATH = PROJECT_ROOT / "outputs" / "index.faiss"

TEST_FOLDER = PROJECT_ROOT / "data" / "test" / "defective"
# TEST_FOLDER = PROJECT_ROOT / "data" / "test" / "good"

RESULTS_FOLDER = PROJECT_ROOT / "outputs" / "results"

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# FIND TEST IMAGE
# ==========================================================

image_paths = []

for ext in IMAGE_EXTENSIONS:
    image_paths.extend(TEST_FOLDER.glob(f"*{ext}"))

image_paths = sorted(image_paths)

if len(image_paths) == 0:
    raise FileNotFoundError(
        f"No images found in:\n{TEST_FOLDER}"
    )

image_path = image_paths[0]

# ==========================================================
# LOAD IMAGE
# ==========================================================

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(
        f"Unable to read image:\n{image_path}"
    )

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# ==========================================================
# LOAD PATCHCORE DETECTOR
# ==========================================================

detector = AnomalyDetector(INDEX_PATH)

# ==========================================================
# RUN PATCHCORE
# ==========================================================

distances, indices, image_score = detector.predict(
    image_rgb
)

# ==========================================================
# IMAGE PREDICTION
# ==========================================================

if image_score >= THRESHOLD:
    prediction = "DEFECTIVE"
    status = "ANOMALY DETECTED"
else:
    prediction = "GOOD"
    status = "NORMAL SURFACE"

# ==========================================================
# GENERATE HEATMAP
# ==========================================================

heatmap_generator = HeatmapGenerator()

anomaly_map, heatmap = heatmap_generator.generate(
    distances,
    image.shape
)

overlay = heatmap_generator.overlay(
    image,
    heatmap
)

# ==========================================================
# SAVE IMAGES
# ==========================================================

cv2.imwrite(
    str(RESULTS_FOLDER / "original.png"),
    image
)

cv2.imwrite(
    str(RESULTS_FOLDER / "anomaly_map.png"),
    anomaly_map
)

cv2.imwrite(
    str(RESULTS_FOLDER / "heatmap.png"),
    heatmap
)

cv2.imwrite(
    str(RESULTS_FOLDER / "overlay.png"),
    overlay
)

# ==========================================================
# GENERATE REPORT
# ==========================================================

report = ReportGenerator()

report.generate(
    original=image,
    heatmap=heatmap,
    overlay=overlay,
    prediction=prediction,
    score=image_score,
    threshold=THRESHOLD,
    save_path=RESULTS_FOLDER / "report.png",
    image_name=image_path.name
)

# ==========================================================
# SAVE PREDICTION DETAILS
# ==========================================================

prediction_file = RESULTS_FOLDER / "prediction.txt"

with open(prediction_file, "w") as f:

    f.write("PatchCore Metal Surface Inspection\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Image      : {image_path.name}\n")
    f.write(f"Prediction : {prediction}\n")
    f.write(f"Status     : {status}\n")
    f.write(f"Score      : {image_score:.2f}\n")
    f.write(f"Threshold  : {THRESHOLD:.2f}\n")

# ==========================================================
# TERMINAL OUTPUT
# ==========================================================

print("\n" + "=" * 60)
print("PatchCore Metal Surface Inspection")
print("=" * 60)

print(f"Image       : {image_path.name}")
print(f"Prediction  : {prediction}")
print(f"Status      : {status}")
print(f"Score       : {image_score:.2f}")
print(f"Threshold   : {THRESHOLD:.2f}")

print("\nSaved Files")
print("-" * 30)

print("✓ original.png")
print("✓ anomaly_map.png")
print("✓ heatmap.png")
print("✓ overlay.png")
print("✓ report.png")
print("✓ prediction.txt")

print()
print(f"Location : {RESULTS_FOLDER}")

print("=" * 60)