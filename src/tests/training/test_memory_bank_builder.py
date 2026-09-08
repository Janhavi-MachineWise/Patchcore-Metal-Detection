from pathlib import Path

from memory_bank_builder import MemoryBankBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FOLDER = PROJECT_ROOT / "data" / "train" / "good"

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "memory_bank.dat"

builder = MemoryBankBuilder()

memory_bank = builder.build(
    TRAIN_FOLDER,
    OUTPUT_FILE
)

print()

print("=" * 60)

print("Memory Bank Shape :", memory_bank.shape)

print("Data Type         :", memory_bank.dtype)

print("=" * 60)