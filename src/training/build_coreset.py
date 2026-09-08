from pathlib import Path
import numpy as np

from random_projection import RandomProjector
from coreset_sampler import ApproximateGreedyCoresetSampler

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEMORY_BANK_PATH = PROJECT_ROOT / "outputs" / "memory_bank.dat"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "coreset.npy"

MEMORY_BANK_SHAPE = (272832, 1536)

SAMPLING_RATIO = 0.05

# ==========================================================
# LOAD MEMORY BANK
# ==========================================================

memory_bank = np.memmap(
    MEMORY_BANK_PATH,
    dtype=np.float32,
    mode="r",
    shape=MEMORY_BANK_SHAPE
)

print("=" * 60)
print("Memory Bank Loaded")
print(memory_bank.shape)
print("=" * 60)

# ==========================================================
# RANDOM PROJECTION
# ==========================================================

projector = RandomProjector()

projected = projector.fit_transform(memory_bank)

print("Projected Shape :", projected.shape)

# ==========================================================
# CORESET SAMPLING
# ==========================================================

sampler = ApproximateGreedyCoresetSampler(
    sampling_ratio=SAMPLING_RATIO
)

indices = sampler.sample(projected)

print()

print("Selected :", len(indices))

# ==========================================================
# BUILD FINAL CORESET
# ==========================================================

coreset = memory_bank[indices]

print("Coreset Shape :", coreset.shape)

# ==========================================================
# SAVE
# ==========================================================

np.save(
    OUTPUT_PATH,
    coreset
)

print()

print("Saved to")

print(OUTPUT_PATH)