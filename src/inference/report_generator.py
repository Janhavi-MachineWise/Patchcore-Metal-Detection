import cv2
import matplotlib.pyplot as plt


class ReportGenerator:

    def __init__(self):
        pass

    def generate(
        self,
        original,
        heatmap,
        overlay,
        prediction,
        score,
        threshold,
        save_path,
        image_name=""
    ):

        original = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        fig = plt.figure(figsize=(16,9))

        fig.suptitle(
            "PatchCore Metal Surface Inspection",
            fontsize=22,
            fontweight="bold"
        )

        # --------------------------------------------------
        # Original
        # --------------------------------------------------

        ax1 = plt.subplot(2,3,1)

        ax1.imshow(original)

        ax1.set_title(
            "Original Image",
            fontsize=14,
            fontweight="bold"
        )

        ax1.axis("off")

        # --------------------------------------------------
        # Heatmap
        # --------------------------------------------------

        ax2 = plt.subplot(2,3,2)

        ax2.imshow(heatmap)

        ax2.set_title(
            "Anomaly Heatmap",
            fontsize=14,
            fontweight="bold"
        )

        ax2.axis("off")

        # --------------------------------------------------
        # Overlay
        # --------------------------------------------------

        ax3 = plt.subplot(2,3,3)

        ax3.imshow(overlay)

        ax3.set_title(
            "Overlay Result",
            fontsize=14,
            fontweight="bold"
        )

        ax3.axis("off")

        # --------------------------------------------------
        # Information Panel
        # --------------------------------------------------

        ax4 = plt.subplot(2,1,2)

        ax4.axis("off")

        if prediction == "DEFECTIVE":

            status_color = "red"

            status_text = "🔴 DEFECT DETECTED"

        else:

            status_color = "green"

            status_text = "🟢 NORMAL SURFACE"

        info = (
            f"Image        : {image_name}\n\n"
            f"Prediction   : {prediction}\n"
            f"Status       : {status_text}\n\n"
            f"Score        : {score:.2f}\n"
            f"Threshold    : {threshold:.2f}\n\n"
            f"Model        : PatchCore\n"
            f"Backbone     : ResNet50\n"
            f"Search       : FAISS"
        )

        ax4.text(
            0.02,
            0.95,
            info,
            fontsize=15,
            family="monospace",
            verticalalignment="top"
        )

        ax4.text(
            0.72,
            0.55,
            status_text,
            fontsize=24,
            color=status_color,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()