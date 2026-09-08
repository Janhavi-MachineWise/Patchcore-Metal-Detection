from pathlib import Path
import cv2
import pandas as pd

from anomaly_detector import AnomalyDetector

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = PROJECT_ROOT / "outputs" / "index.faiss"

# Change this to "defective" to evaluate defective images
TEST_CATEGORY = "defective"

TEST_FOLDER = PROJECT_ROOT / "data" / "test" / TEST_CATEGORY

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# ==========================================================
# LOAD DETECTOR
# ==========================================================

detector = AnomalyDetector(INDEX_PATH)

# ==========================================================
# FIND IMAGES
# ==========================================================

image_paths = []

for ext in IMAGE_EXTENSIONS:
    image_paths.extend(TEST_FOLDER.glob(f"*{ext}"))

image_paths = sorted(image_paths)

print("=" * 70)
print(f"Evaluating {len(image_paths)} {TEST_CATEGORY} images")
print("=" * 70)

results = []

# ==========================================================
# INFERENCE
# ==========================================================

for image_path in image_paths:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    distances, _ = detector.predict(image)

    results.append({
        "Image": image_path.name,
        "Min": float(distances.min()),
        "Mean": float(distances.mean()),
        "Max": float(distances.max())
    })

# ==========================================================
# RESULTS
# ==========================================================

df = pd.DataFrame(results)

print(df)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("Images :", len(df))
print("Average Mean Distance :", df["Mean"].mean())
print("Average Max Distance  :", df["Max"].mean())

# ==========================================================
# SAVE CSV
# ==========================================================

output_path = PROJECT_ROOT / "outputs" / f"{TEST_CATEGORY}_scores.csv"

df.to_csv(output_path, index=False)

print()
print("Saved:", output_path)