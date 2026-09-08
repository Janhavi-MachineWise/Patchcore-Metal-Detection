from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from config import PATCH_SIZE, STRIDE

# ==========================================================
# IMAGE PATH
# ==========================================================

IMAGE_PATH = PROJECT_ROOT / "data" / "resized" / "good"

image_path = sorted(IMAGE_PATH.glob("*"))[0]

# ==========================================================
# LOAD IMAGE
# ==========================================================

image = cv2.imread(str(image_path))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

patches = []

positions = []

H, W = image.shape[:2]

# ==========================================================
# GENERATE PATCHES
# ==========================================================

for y in range(0, H - PATCH_SIZE + 1, STRIDE):

    for x in range(0, W - PATCH_SIZE + 1, STRIDE):

        patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

        patches.append(patch)

        positions.append((x, y))

# ==========================================================
# RESULTS
# ==========================================================

print("=" * 50)

print(f"Total Patches : {len(patches)}")

print(f"First Coordinate : {positions[0]}")

print(f"Last Coordinate : {positions[-1]}")

print("=" * 50)

# ==========================================================
# VISUALIZE GRID
# ==========================================================

image_vis = image.copy()

for x, y in positions:

    cv2.rectangle(
        image_vis,
        (x, y),
        (x + PATCH_SIZE, y + PATCH_SIZE),
        (255, 0, 0),
        2
    )

plt.figure(figsize=(8, 8))
plt.imshow(image_vis)
plt.title("Patch Grid")
plt.axis("off")
plt.show()