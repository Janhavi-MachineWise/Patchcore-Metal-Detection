import sys
from pathlib import Path

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from config import MODEL_INPUT_SIZE


class FeatureExtractor:

    def __init__(self):

        print("Loading PatchCore Feature Extractor...")

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Device:", self.device)

        # --------------------------------------------------
        # Load pretrained ResNet50
        # --------------------------------------------------

        self.model = models.resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        self.model.eval()
        self.model.to(self.device)

        # --------------------------------------------------
        # Store intermediate feature maps
        # --------------------------------------------------

        self.features = {}

        self.model.layer2.register_forward_hook(
            self.save_layer2
        )

        self.model.layer3.register_forward_hook(
            self.save_layer3
        )

        # --------------------------------------------------
        # Image preprocessing
        # --------------------------------------------------

        self.transform = transforms.Compose([
            transforms.ToPILImage(),

            transforms.Resize(
                (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Feature extractor ready.")

    # --------------------------------------------------
    # Hook functions
    # --------------------------------------------------

    def save_layer2(self, module, input, output):
        self.features["layer2"] = output.detach()

    def save_layer3(self, module, input, output):
        self.features["layer3"] = output.detach()

    # --------------------------------------------------
    # Feature Extraction (Single Image)
    # --------------------------------------------------

    def extract_features(self, image):
        """
        Extract Layer2 and Layer3 feature maps
        from one RGB image.

        Parameters
        ----------
        image : numpy.ndarray (RGB)

        Returns
        -------
        layer2 : torch.Tensor
            Shape -> [1,512,28,28]

        layer3 : torch.Tensor
            Shape -> [1,1024,14,14]
        """

        image = self.transform(image)

        image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            _ = self.model(image)

        return (
            self.features["layer2"].cpu(),
            self.features["layer3"].cpu()
        )

    # --------------------------------------------------
    # Feature Extraction (Batch)
    # --------------------------------------------------

    def extract_features_batch(self, patches):
        """
        Extract Layer2 and Layer3 feature maps
        from a batch of RGB patches.

        Parameters
        ----------
        patches : list
            List of RGB numpy images.

        Returns
        -------
        layer2 : torch.Tensor
            Shape -> [B,512,28,28]

        layer3 : torch.Tensor
            Shape -> [B,1024,14,14]
        """

        batch = []

        for patch in patches:
            batch.append(self.transform(patch))

        batch = torch.stack(batch)

        batch = batch.to(self.device)

        with torch.no_grad():
            _ = self.model(batch)

        return (
            self.features["layer2"].cpu(),
            self.features["layer3"].cpu()
        )