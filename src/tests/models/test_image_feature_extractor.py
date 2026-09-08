from pathlib import Path
import cv2

from feature_extractor import FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "train" / "good"

image_path = sorted(IMAGE_PATH.glob("*"))[0]

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

extractor = FeatureExtractor()

layer2, layer3 = extractor.extract_features(image)

print("=" * 60)
print("Layer2 :", layer2.shape)
print("Layer3 :", layer3.shape)
print("=" * 60)