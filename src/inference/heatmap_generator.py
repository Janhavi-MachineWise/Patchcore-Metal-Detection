import cv2
import numpy as np


class HeatmapGenerator:
    """
    Generate PatchCore anomaly heatmaps and overlay them
    on the original image.
    """

    def __init__(
        self,
        heatmap_size=(28, 28),
        overlay_alpha=0.65,
        heatmap_alpha=0.35
    ):

        self.heatmap_size = heatmap_size
        self.overlay_alpha = overlay_alpha
        self.heatmap_alpha = heatmap_alpha

    # -------------------------------------------------------
    # Generate Heatmap
    # -------------------------------------------------------

    def generate(self, distances, image_shape):
        """
        Parameters
        ----------
        distances : ndarray
            Shape -> (784,)

        image_shape : tuple
            Original image shape (H, W)

        Returns
        -------
        anomaly_map : ndarray

        colored_heatmap : ndarray
        """

        # ------------------------------------------
        # 784 -> 28x28
        # ------------------------------------------

        anomaly_map = distances.reshape(self.heatmap_size)

        # ------------------------------------------
        # Normalize to [0,255]
        # ------------------------------------------

        anomaly_map = cv2.normalize(
            anomaly_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        anomaly_map = anomaly_map.astype(np.uint8)

        # ------------------------------------------
        # Smooth anomaly map
        # ------------------------------------------

        anomaly_map = cv2.GaussianBlur(
            anomaly_map,
            (5, 5),
            sigmaX=0
        )

        # ------------------------------------------
        # Resize
        # ------------------------------------------

        H, W = image_shape[:2]

        anomaly_map = cv2.resize(
            anomaly_map,
            (W, H),
            interpolation=cv2.INTER_CUBIC
        )

        # ------------------------------------------
        # Normalize again after resize
        # ------------------------------------------

        anomaly_map = cv2.normalize(
            anomaly_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        anomaly_map = anomaly_map.astype(np.uint8)

        # ------------------------------------------
        # Apply Color Map
        # ------------------------------------------

        colored_heatmap = cv2.applyColorMap(
            anomaly_map,
            cv2.COLORMAP_JET
        )

        return anomaly_map, colored_heatmap

    # -------------------------------------------------------
    # Overlay Heatmap
    # -------------------------------------------------------

    def overlay(self, image, heatmap):
        """
        Overlay heatmap on original image.
        """

        overlay = cv2.addWeighted(
            image,
            self.overlay_alpha,
            heatmap,
            self.heatmap_alpha,
            0
        )

        return overlay