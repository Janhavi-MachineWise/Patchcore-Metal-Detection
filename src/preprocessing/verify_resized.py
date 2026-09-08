from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = PROJECT_ROOT / "data" / "resized" / "good"

wrong = []

for image_path in sorted(DATASET.glob("*")):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]

    if (w, h) != (1024, 1024):
        wrong.append((image_path.name, w, h))

print("=" * 50)
print(f"Total Images : {len(list(DATASET.glob('*')))}")
print(f"Wrong Size   : {len(wrong)}")
print("=" * 50)

if wrong:
    for item in wrong:
        print(item)
else:
    print("All images are 1024 × 1024 ✅")