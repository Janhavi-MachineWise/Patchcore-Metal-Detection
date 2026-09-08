import sys
from pathlib import Path

from sklearn.random_projection import SparseRandomProjection

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from config import PROJECTION_DIM


class RandomProjector:
    """
    Reduce descriptor dimensionality before
    Approximate Greedy Coreset Sampling.
    """

    def __init__(self, output_dim=PROJECTION_DIM):

        self.output_dim = output_dim

        self.projector = SparseRandomProjection(
            n_components=self.output_dim,
            random_state=42
        )

    def fit_transform(self, memory_bank):

        projected = self.projector.fit_transform(
            memory_bank
        )

        return projected