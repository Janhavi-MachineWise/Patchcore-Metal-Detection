import torch
import numpy as np


class ApproximateGreedyCoresetSampler:

    def __init__(self, sampling_ratio=0.10, device=None):

        self.sampling_ratio = sampling_ratio

        self.device = (
            device if device
            else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    # ----------------------------------------------------
    # Convert numpy → torch
    # ----------------------------------------------------

    def to_tensor(self, features):

        if isinstance(features, np.ndarray):

            features = torch.from_numpy(features)

        return features.float().to(self.device)

    # ----------------------------------------------------
    # Compute Euclidean Distance
    # ----------------------------------------------------

    def compute_distances(self, point, features):

        return torch.norm(
            features - point,
            dim=1
        )

    # ----------------------------------------------------
    # Greedy Sampling
    # ----------------------------------------------------

    def sample(self, features):

        features = self.to_tensor(features)

        n_samples = int(
            len(features) * self.sampling_ratio
        )

        selected_indices = []

        # Random starting point
        current_index = torch.randint(
            len(features),
            (1,)
        ).item()

        selected_indices.append(current_index)

        min_distances = self.compute_distances(
            features[current_index],
            features
        )

        while len(selected_indices) < n_samples:

            current_index = torch.argmax(
                min_distances
            ).item()

            selected_indices.append(current_index)

            new_distances = self.compute_distances(
                features[current_index],
                features
            )

            min_distances = torch.minimum(
                min_distances,
                new_distances
            )

        return np.array(selected_indices)