from pathlib import Path
import numpy as np

from random_projection import RandomProjector
from coreset_sampler import ApproximateGreedyCoresetSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------
# Load Memory Bank
# --------------------------------------------------

memory_bank = np.memmap(
    PROJECT_ROOT / "outputs" / "memory_bank.dat",
    dtype=np.float32,
    mode="r",
    shape=(272832, 1536)
)

# --------------------------------------------------
# Random Projection
# --------------------------------------------------

projector = RandomProjector()

projected = projector.fit_transform(memory_bank)

print("Projected Shape :", projected.shape)

# --------------------------------------------------
# Coreset Sampler
# --------------------------------------------------

sampler = ApproximateGreedyCoresetSampler()

point = projected[0]

distances = sampler.compute_distances(
    point,
    projected
)

print()

print("Distance Vector Shape :", distances.shape)

print("Minimum Distance :", distances.min())

print("Maximum Distance :", distances.max())