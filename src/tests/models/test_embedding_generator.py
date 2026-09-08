from pathlib import Path
import cv2

from feature_extractor import FeatureExtractor
from embedding_generator import EmbeddingGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = sorted(
    (PROJECT_ROOT / "data" / "resized" / "good").glob("*")
)[0]

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

extractor = FeatureExtractor()

layer2, layer3 = extractor.extract_features(image)

generator = EmbeddingGenerator()

embeddings = generator.generate(
    layer2,
    layer3
)

print()

print("Embedding Shape :", embeddings.shape)

print()

print("One embedding dimension :", embeddings.shape[-1])

print("Total embeddings :", embeddings.shape[1])