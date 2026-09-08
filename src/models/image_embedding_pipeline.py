from pathlib import Path
import sys

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# ==========================================================
# IMPORTS
# ==========================================================

from models.feature_extractor import FeatureExtractor
from models.embedding_generator import EmbeddingGenerator


# ==========================================================
# IMAGE EMBEDDING PIPELINE
# ==========================================================

class ImageEmbeddingPipeline:

    def __init__(self):

        self.feature_extractor = FeatureExtractor()

        self.embedding_generator = EmbeddingGenerator()

    def process(self, image):
        """
        Convert one RGB image into PatchCore embeddings.

        Parameters
        ----------
        image : numpy.ndarray (RGB)

        Returns
        -------
        torch.Tensor
            Shape -> [784,1536]
        """

        # ------------------------------------------
        # Feature Extraction
        # ------------------------------------------

        layer2, layer3 = self.feature_extractor.extract_features(image)

        # ------------------------------------------
        # Embedding Generation
        # ------------------------------------------

        embeddings = self.embedding_generator.generate(
            layer2,
            layer3
        )

        # Remove batch dimension
        embeddings = embeddings.squeeze(0)

        return embeddings