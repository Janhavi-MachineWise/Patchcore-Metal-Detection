from pathlib import Path
import sys
import cv2
import streamlit as st
import numpy as np

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# ==========================================================
# IMPORTS
# ==========================================================

from inference.anomaly_detector import AnomalyDetector
from inference.heatmap_generator import HeatmapGenerator
from config import THRESHOLD

# ==========================================================
# CONFIGURATION
# ==========================================================

INDEX_PATH = PROJECT_ROOT / "outputs" / "index.faiss"

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="PatchCore Metal Inspection",
    page_icon="🔍",
    layout="wide"
)

# Reduce page padding
st.markdown("""
<style>
.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    padding-left:2rem;
    padding-right:2rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.title("🔍 PatchCore Metal Surface Inspection")
st.caption("Upload a metal surface image to detect anomalies.")

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_detector():
    return AnomalyDetector(INDEX_PATH)

detector = load_detector()

# ==========================================================
# INFERENCE
# ==========================================================

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------------------
    # Run PatchCore
    # ------------------------------------------------------

    with st.spinner("Running PatchCore..."):

        distances, indices, image_score = detector.predict(
            image_rgb
        )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    if image_score >= THRESHOLD:
        prediction = "DEFECTIVE 🔴"
    else:
        prediction = "GOOD 🟢"

    # ------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------

    heatmap_generator = HeatmapGenerator()

    anomaly_map, heatmap = heatmap_generator.generate(
        distances,
        image.shape
    )

    overlay = heatmap_generator.overlay(
        image,
        heatmap
    )

    # ======================================================
    # METRICS
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", prediction)

    with col2:
        st.metric("Score", f"{image_score:.2f}")

    with col3:
        st.metric("Threshold", f"{THRESHOLD:.2f}")

    st.divider()

    # ======================================================
    # IMAGES
    # ======================================================

    IMAGE_WIDTH = 240

    left, col1, col2, col3, right = st.columns(
        [0.3, 1, 1, 1, 0.3]
    )

    with col1:

        st.image(
            image_rgb,
            caption="Original",
            width=IMAGE_WIDTH
        )

    with col2:

        st.image(
            cv2.cvtColor(
                heatmap,
                cv2.COLOR_BGR2RGB
            ),
            caption="Heatmap",
            width=IMAGE_WIDTH
        )

    with col3:

        st.image(
            cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            ),
            caption="Overlay",
            width=IMAGE_WIDTH
        )

    st.divider()

    # ======================================================
    # DOWNLOAD REPORT
    # ======================================================

    report_path = PROJECT_ROOT / "outputs" / "results" / "report.png"

    if report_path.exists():

        left, center, right = st.columns([1, 2, 1])

        with center:

            with open(report_path, "rb") as file:

                st.download_button(
                    label="📥 Download Inspection Report",
                    data=file,
                    file_name="PatchCore_Report.png",
                    mime="image/png",
                    use_container_width=True
                )