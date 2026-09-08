from pathlib import Path

from roi_extractor import ROIExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

extractor = ROIExtractor(crop_ratio=0.85)

# --------------------------
# GOOD
# --------------------------

extractor.process_folder(
    PROJECT_ROOT / "data/original/good",
    PROJECT_ROOT / "data/roi/good",
)

# --------------------------
# DEFECTIVE
# --------------------------

extractor.process_folder(
    PROJECT_ROOT / "data/original/defective",
    PROJECT_ROOT / "data/roi/defective",
)