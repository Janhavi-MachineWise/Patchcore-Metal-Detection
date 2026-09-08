from pathlib import Path
from collections import Counter

import cv2
from tqdm import tqdm


class DatasetAnalyzer:
    """
    Analyze an image dataset and generate basic statistics.
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def get_images(self, folder_name):
        """
        Get all image paths from a folder.
        """

        folder = self.dataset_path / folder_name

        image_extensions = [".jpg", ".jpeg", ".png"]

        image_paths = []

        for extension in image_extensions:
            image_paths.extend(folder.glob(f"*{extension}"))

        return sorted(image_paths)

    def analyze_images(self, image_paths):
        """
        Analyze a list of images and collect statistics.
        """

        total_images = len(image_paths)

        corrupted_images = 0
        rgb_images = 0
        grayscale_images = 0

        width_counter = Counter()
        height_counter = Counter()

        for image_path in tqdm(image_paths, desc="Analyzing Images"):

            image = cv2.imread(str(image_path))

            if image is None:
                corrupted_images += 1
                continue

            height, width = image.shape[:2]

            width_counter[width] += 1
            height_counter[height] += 1

            if len(image.shape) == 3:
                rgb_images += 1
            else:
                grayscale_images += 1

        return {
            "total_images": total_images,
            "corrupted_images": corrupted_images,
            "rgb_images": rgb_images,
            "grayscale_images": grayscale_images,
            "width_counter": width_counter,
            "height_counter": height_counter,
        }

    def print_report(self, folder_name, stats):
        """
        Print dataset statistics.
        """

        print("\n" + "=" * 50)
        print(f"DATASET REPORT : {folder_name.upper()}")
        print("=" * 50)

        print(f"Total Images      : {stats['total_images']}")
        print(f"Corrupted Images  : {stats['corrupted_images']}")
        print(f"RGB Images        : {stats['rgb_images']}")
        print(f"Grayscale Images  : {stats['grayscale_images']}")

        print("\nImage Width Distribution")
        print(stats["width_counter"])

        print("\nImage Height Distribution")
        print(stats["height_counter"])

        print("=" * 50)


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    dataset_path = PROJECT_ROOT / "data" / "original"

    analyzer = DatasetAnalyzer(dataset_path)

    # Analyze GOOD images
    good_images = analyzer.get_images("good")
    good_stats = analyzer.analyze_images(good_images)
    analyzer.print_report("good", good_stats)

    # Analyze DEFECTIVE images
    defective_images = analyzer.get_images("defective")
    defective_stats = analyzer.analyze_images(defective_images)
    analyzer.print_report("defective", defective_stats)


if __name__ == "__main__":
    main()