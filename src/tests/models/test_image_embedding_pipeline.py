from pathlib import Path
import cv2

from image_embedding_pipeline import ImageEmbeddingPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_FOLDER = PROJECT_ROOT / "data" / "train" / "good"

image_path = sorted(IMAGE_FOLDER.glob("*"))[0]

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pipeline = ImageEmbeddingPipeline()

embeddings = pipeline.process(image)

print("=" * 60)

print("Embedding Shape :", embeddings.shape)

print("Descriptors :", embeddings.shape[0])

print("Embedding Dimension :", embeddings.shape[1])

print("=" * 60)

