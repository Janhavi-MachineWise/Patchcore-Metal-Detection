from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(str(SRC_PATH))

from models.image_embedding_pipeline import ImageEmbeddingPipeline


class MemoryBankBuilder:

    def __init__(self):

        self.pipeline = ImageEmbeddingPipeline()

    def build(self, image_folder, output_file):

        image_paths = sorted(Path(image_folder).glob("*"))

        total_images = len(image_paths)

        descriptors_per_image = 784

        embedding_dim = 1536

        total_descriptors = total_images * descriptors_per_image

        print("=" * 60)
        print(f"Training Images      : {total_images}")
        print(f"Total Descriptors    : {total_descriptors}")
        print("=" * 60)

        # --------------------------------------------------
        # Create memory-mapped file
        # --------------------------------------------------

        memory_bank = np.memmap(
            output_file,
            dtype=np.float32,
            mode="w+",
            shape=(total_descriptors, embedding_dim)
        )

        current_index = 0

        for image_path in tqdm(image_paths, desc="Building Memory Bank"):

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            embeddings = self.pipeline.process(image)

            # Torch → NumPy
            embeddings = embeddings.numpy().astype(np.float32)

            rows = embeddings.shape[0]

            memory_bank[
                current_index:current_index + rows
            ] = embeddings

            current_index += rows

        memory_bank.flush()

        return memory_bank