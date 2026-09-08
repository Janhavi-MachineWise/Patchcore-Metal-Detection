from pathlib import Path
import sys
import numpy as np
import faiss

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

from models.image_embedding_pipeline import ImageEmbeddingPipeline


# ==========================================================
# ANOMALY DETECTOR
# ==========================================================

class AnomalyDetector:

    def __init__(self, index_path):
        """
        Load PatchCore embedding pipeline
        and FAISS index.
        """

        print("Loading PatchCore Inference Pipeline...")

        self.pipeline = ImageEmbeddingPipeline()

        self.index = faiss.read_index(str(index_path))

        print("FAISS index loaded successfully.")
        print("Total Normal Descriptors :", self.index.ntotal)

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    def predict(self, image):
        """
        Predict anomaly score for one RGB image.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image

        Returns
        -------
        distances : ndarray
            Euclidean distances
            Shape -> (784,)

        indices : ndarray
            Shape -> (784,)

        image_score : float
            Maximum Euclidean anomaly score
        """

        # ----------------------------------------------
        # Generate PatchCore embeddings
        # ----------------------------------------------

        embeddings = self.pipeline.process(image)

        embeddings = embeddings.cpu().numpy().astype(np.float32)

        # ----------------------------------------------
        # FAISS Search
        # Returns Squared L2 Distances
        # ----------------------------------------------

        distances, indices = self.index.search(
            embeddings,
            k=1
        )

        # ----------------------------------------------
        # Convert Squared L2 -> Euclidean Distance
        # ----------------------------------------------

        distances = np.sqrt(distances)

        # Remove extra dimension
        distances = distances.squeeze(1)
        indices = indices.squeeze(1)

        # ----------------------------------------------
        # Image-level anomaly score
        # ----------------------------------------------

        image_score = float(np.max(distances))

        return distances, indices, image_score