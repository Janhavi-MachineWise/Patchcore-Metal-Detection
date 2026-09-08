from pathlib import Path
import sys
import cv2
from tqdm import tqdm

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from config import IMAGE_SIZE

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FOLDER = PROJECT_ROOT / "data" / "roi" / "good"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "resized" / "good"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ==========================================================
# LOAD IMAGES
# ==========================================================

image_paths = []

for ext in IMAGE_EXTENSIONS:
    image_paths.extend(INPUT_FOLDER.glob(f"*{ext}"))

image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images.")

# ==========================================================
# RESIZE
# ==========================================================

for image_path in tqdm(image_paths):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    resized = cv2.resize(
        image,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA
    )

    save_path = OUTPUT_FOLDER / image_path.name

    cv2.imwrite(str(save_path), resized)

print("\nFinished resizing dataset.")