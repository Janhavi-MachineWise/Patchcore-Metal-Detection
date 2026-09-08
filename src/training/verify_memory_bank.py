from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

memory_bank = np.memmap(
    PROJECT_ROOT / "outputs" / "memory_bank.dat",
    dtype=np.float32,
    mode="r",
    shape=(272832, 1536)
)

print("=" * 60)

print("Shape :", memory_bank.shape)

print("Dtype :", memory_bank.dtype)

print()

print("First Descriptor (first 10 values):")
print(memory_bank[0][:10])

print()

print("Descriptor Mean :", memory_bank.mean())

print("Descriptor Std  :", memory_bank.std())

print("=" * 60)