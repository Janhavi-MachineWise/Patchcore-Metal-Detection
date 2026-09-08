from pathlib import Path
import cv2

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = PROJECT_ROOT / "data" / "original" / "good"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "roi" / "good"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_DISPLAY_WIDTH = 1400

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ==========================================================
# LOAD IMAGE LIST
# ==========================================================

image_paths = []

for ext in IMAGE_EXTENSIONS:
    image_paths.extend(INPUT_FOLDER.glob(f"*{ext}"))

image_paths = sorted(image_paths)

print(f"\nFound {len(image_paths)} images.")

# ==========================================================
# MAIN LOOP
# ==========================================================

for index, image_path in enumerate(image_paths):

    save_path = OUTPUT_FOLDER / image_path.name

    if save_path.exists():
        print(f"[{index+1}/{len(image_paths)}] Already Done")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]

    # ---------------------------------------------
    # Resize only for DISPLAY
    # ---------------------------------------------

    scale = 1.0

    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w

    display = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    print("\n" + "=" * 60)
    print(f"{index+1}/{len(image_paths)}")
    print(image_path.name)
    print("=" * 60)

    print("""
Draw ROI

ENTER / SPACE -> Save

Cancel ROI -> Skip
""")

    roi = cv2.selectROI(
        "Crop Tool",
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("Crop Tool")

    x, y, rw, rh = roi

    # ---------------------------------------------
    # Skip
    # ---------------------------------------------

    if rw == 0 or rh == 0:
        print("Skipped")
        continue

    # ---------------------------------------------
    # Convert coordinates back
    # ---------------------------------------------

    x = int(x / scale)
    y = int(y / scale)

    rw = int(rw / scale)
    rh = int(rh / scale)

    crop = image[y:y+rh, x:x+rw]

    cv2.imwrite(str(save_path), crop)

    print("Saved")

print("\nFinished!")

cv2.destroyAllWindows()