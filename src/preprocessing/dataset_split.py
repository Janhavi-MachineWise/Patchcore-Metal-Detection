from pathlib import Path
import shutil
import random

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOOD_SOURCE = PROJECT_ROOT / "data" / "resized" / "good"

DEFECTIVE_SOURCE = PROJECT_ROOT / "data" / "original" / "defective"

TRAIN_GOOD = PROJECT_ROOT / "data" / "train" / "good"

TEST_GOOD = PROJECT_ROOT / "data" / "test" / "good"

TEST_DEFECTIVE = PROJECT_ROOT / "data" / "test" / "defective"

TRAIN_RATIO = 0.80

SEED = 42

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

TRAIN_GOOD.mkdir(parents=True, exist_ok=True)
TEST_GOOD.mkdir(parents=True, exist_ok=True)
TEST_DEFECTIVE.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD GOOD IMAGES
# ==========================================================

image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(GOOD_SOURCE.glob(ext))

image_paths = sorted(image_paths)

print(f"Total Good Images : {len(image_paths)}")

# ==========================================================
# SHUFFLE
# ==========================================================

random.seed(SEED)

random.shuffle(image_paths)

# ==========================================================
# SPLIT
# ==========================================================

split_index = int(len(image_paths) * TRAIN_RATIO)

train_images = image_paths[:split_index]

test_images = image_paths[split_index:]

# ==========================================================
# COPY TRAIN
# ==========================================================

for img in train_images:

    shutil.copy2(
        img,
        TRAIN_GOOD / img.name
    )

# ==========================================================
# COPY TEST GOOD
# ==========================================================

for img in test_images:

    shutil.copy2(
        img,
        TEST_GOOD / img.name
    )

# ==========================================================
# COPY DEFECTIVE
# ==========================================================

defective_images = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    defective_images.extend(DEFECTIVE_SOURCE.glob(ext))

for img in defective_images:

    shutil.copy2(
        img,
        TEST_DEFECTIVE / img.name
    )

# ==========================================================
# REPORT
# ==========================================================

print()
print("=" * 50)

print("Training Good :", len(train_images))

print("Testing Good  :", len(test_images))

print("Testing Defective :", len(defective_images))

print("=" * 50)