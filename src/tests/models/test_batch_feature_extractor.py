from pathlib import Path
import cv2

from feature_extractor import FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = sorted(
    (PROJECT_ROOT / "data" / "train" / "good").glob("*")
)[0]

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create dummy batch of 4 patches
patches = [image, image, image, image]

extractor = FeatureExtractor()

layer2, layer3 = extractor.extract_features_batch(patches)

print(layer2.shape)
print(layer3.shape)