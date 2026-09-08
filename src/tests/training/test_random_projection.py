from pathlib import Path
import numpy as np

from random_projection import RandomProjector

PROJECT_ROOT = Path(__file__).resolve().parents[2]

memory_bank = np.memmap(
    PROJECT_ROOT / "outputs" / "memory_bank.dat",
    dtype=np.float32,
    mode="r",
    shape=(272832,1536)
)

print("Original Shape :", memory_bank.shape)

projector = RandomProjector()

projected = projector.fit_transform(memory_bank)

print("Projected Shape :", projected.shape)