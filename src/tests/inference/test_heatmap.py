from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from anomaly_detector import AnomalyDetector
from heatmap_generator import HeatmapGenerator

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "outputs" / "index.faiss"

TEST_CATEGORY = "defective"
# TEST_CATEGORY = "good"

TEST_FOLDER = PROJECT_ROOT / "data" / "test" / TEST_CATEGORY

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# ==========================================================
# FIND IMAGE
# ==========================================================

image_paths = []

for ext in IMAGE_EXTENSIONS:
    image_paths.extend(TEST_FOLDER.glob(f"*{ext}"))

image_paths = sorted(image_paths)

image_path = image_paths[0]

print("Testing:", image_path.name)

# ==========================================================
# LOAD IMAGE
# ==========================================================

image = cv2.imread(str(image_path))

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# ==========================================================
# DETECTOR
# ==========================================================

detector = AnomalyDetector(INDEX_PATH)

distances, _ = detector.predict(image_rgb)

# ==========================================================
# HEATMAP
# ==========================================================

generator = HeatmapGenerator()

anomaly_map, heatmap = generator.generate(
    distances,
    image.shape
)

overlay = generator.overlay(
    image,
    heatmap
)

overlay = cv2.cvtColor(
    overlay,
    cv2.COLOR_BGR2RGB
)

# ==========================================================
# VISUALIZE
# ==========================================================

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(anomaly_map, cmap="jet")
plt.title("Anomaly Map")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Heatmap Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()