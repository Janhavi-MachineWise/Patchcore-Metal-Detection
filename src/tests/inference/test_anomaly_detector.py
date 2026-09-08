from pathlib import Path
import cv2

from anomaly_detector import AnomalyDetector

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "outputs" / "index.faiss"

# ==========================================================
# SELECT TEST CATEGORY
# ==========================================================

TEST_CATEGORY = "good"
# TEST_CATEGORY = "defective"

TEST_FOLDER = PROJECT_ROOT / "data" / "test" / TEST_CATEGORY

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# ==========================================================
# FIND TEST IMAGES
# ==========================================================

image_paths = []

for ext in IMAGE_EXTENSIONS:
    image_paths.extend(TEST_FOLDER.glob(f"*{ext}"))

image_paths = sorted(image_paths)

if len(image_paths) == 0:
    raise FileNotFoundError(
        f"No images found in:\n{TEST_FOLDER}"
    )

print("=" * 60)
print(f"Test Category : {TEST_CATEGORY}")
print(f"Found {len(image_paths)} test images")
print("=" * 60)

# ==========================================================
# LOAD FIRST IMAGE
# ==========================================================

image_path = image_paths[0]

print(f"Testing Image : {image_path.name}")

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(
        f"Could not read image:\n{image_path}"
    )

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==========================================================
# LOAD ANOMALY DETECTOR
# ==========================================================

detector = AnomalyDetector(INDEX_PATH)

# ==========================================================
# RUN INFERENCE
# ==========================================================

distances, indices = detector.predict(image)

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print()
print("=" * 60)

print("Distances Shape :", distances.shape)
print("Indices Shape   :", indices.shape)

print()

print("Minimum Distance :", distances.min())
print("Maximum Distance :", distances.max())
print("Mean Distance    :", distances.mean())

print("=" * 60)