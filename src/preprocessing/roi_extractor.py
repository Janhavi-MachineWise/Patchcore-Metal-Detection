from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Load Image
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = PROJECT_ROOT / "data" / "original" / "good" / "IMG_20260619_141611_736.jpg"

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"Cannot load image:\n{image_path}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =====================================================
# Grayscale
# =====================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =====================================================
# Blur
# =====================================================

blur = cv2.GaussianBlur(gray, (5,5), 0)

# =====================================================
# OTSU Threshold
# =====================================================

_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# =====================================================
# Morphological Closing
# =====================================================

kernel = np.ones((9,9), np.uint8)

closed = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# =====================================================
# Find Largest Contour
# =====================================================

contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = rgb.copy()

if len(contours) > 0:

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    cv2.rectangle(
        result,
        (x, y),
        (x+w, y+h),
        (255, 0, 0),
        5
    )

    cv2.drawContours(
        result,
        [largest],
        -1,
        (0,255,0),
        3
    )

# =====================================================
# Display
# =====================================================

fig, ax = plt.subplots(2,3, figsize=(18,10))

ax[0,0].imshow(rgb)
ax[0,0].set_title("Original")
ax[0,0].axis("off")

ax[0,1].imshow(gray, cmap="gray")
ax[0,1].set_title("Gray")
ax[0,1].axis("off")

ax[0,2].imshow(blur, cmap="gray")
ax[0,2].set_title("Blur")
ax[0,2].axis("off")

ax[1,0].imshow(thresh, cmap="gray")
ax[1,0].set_title("Otsu Threshold")
ax[1,0].axis("off")

ax[1,1].imshow(closed, cmap="gray")
ax[1,1].set_title("Morphology")
ax[1,1].axis("off")

ax[1,2].imshow(result)
ax[1,2].set_title("Largest Contour")
ax[1,2].axis("off")

plt.tight_layout()
plt.show()