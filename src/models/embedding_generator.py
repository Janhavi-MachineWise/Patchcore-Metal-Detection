import torch
import torch.nn.functional as F


class EmbeddingGenerator:

    def __init__(self):
        pass

    def generate(self, layer2, layer3):
        """
        Generate PatchCore embeddings with local neighborhood aggregation.

        Parameters
        ----------
        layer2 : Tensor
            Shape -> [B, 512, 28, 28]

        layer3 : Tensor
            Shape -> [B, 1024, 14, 14]

        Returns
        -------
        Tensor
            Shape -> [B, 784, 1536]
        """

        # --------------------------------------------------
        # Upsample Layer3
        # --------------------------------------------------

        layer3 = F.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # --------------------------------------------------
        # Concatenate Layer2 and Layer3
        # --------------------------------------------------

        embedding = torch.cat(
            [layer2, layer3],
            dim=1
        )

        # Shape:
        # [B, 1536, 28, 28]

        # --------------------------------------------------
        # Local Neighborhood Aggregation
        # --------------------------------------------------

        embedding = F.avg_pool2d(
            embedding,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Shape:
        # [B, 1536, 28, 28]

        # --------------------------------------------------
        # Flatten Spatial Locations
        # --------------------------------------------------

        B, C, H, W = embedding.shape

        embedding = embedding.permute(
            0, 2, 3, 1
        )

        # Shape:
        # [B, 28, 28, 1536]

        embedding = embedding.reshape(
            B,
            H * W,
            C
        )

        # Shape:
        # [B, 784, 1536]

        return embedding