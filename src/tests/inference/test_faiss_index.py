from pathlib import Path
import numpy as np

from faiss_index import FaissIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORESET_PATH = PROJECT_ROOT / "outputs" / "coreset.npy"

INDEX_PATH = PROJECT_ROOT / "outputs" / "index.faiss"

# ----------------------------------------
# Load Coreset
# ----------------------------------------

coreset = np.load(CORESET_PATH)

print("Coreset Shape :", coreset.shape)

# ----------------------------------------
# Build Index
# ----------------------------------------

builder = FaissIndex()

builder.build(coreset)

builder.save(INDEX_PATH)

print()

print("Saved Index")

print(INDEX_PATH)