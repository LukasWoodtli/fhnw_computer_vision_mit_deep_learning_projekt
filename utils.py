import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import json

from torch.utils.data import Dataset

SCRIPT_DIR = Path(__file__).parent
label_names = json.load(open(SCRIPT_DIR / "part_names.json", "r"))
label_name_to_number = dict((reversed(item) for item in label_names.items()))


def verify_image(fn):
    """Confirm that `fn` can be opened (taken from fast.ai)"""
    try:
        im = Image.open(fn)
        im.draft(im.mode, (32,32))
        im.load()
        return True
    except:
        print(f"File {fn} could not be opened")
        return False


def find_all_images_and_labels(image_dir: str) -> list[dict]:
    """
    Load image paths and corresponding labels.

    Args:
        image_dir: Directory with all the images.

    Returns:
        A list of dicts, one for each obsevation
    """
    observations = list()
    dirs = Path(image_dir).iterdir()
    dirs = [d for d in dirs if d.is_dir()]

    image_extensions = {".jpg", ".jpeg", ".png"}

    for dir in dirs:
        label = dir.name
        class_dir = os.path.join(image_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            verify_image(img_path)
            if any(img_path.lower().endswith(ext) for ext in image_extensions):
                observation = {"image_path": img_path, "label": label_names[label]}
                observations.append(observation)
    return observations


class ImageFolder(Dataset):
    """Create Dataset from class specific folders."""

    def __init__(
        self,
        root_path: str | Path,
        transform: Callable | None = None,
    ):
        """
        Args:
            root_path: Path to directory that contains the class-specific folders
            transform: Optional transform to be applied on an image
            classes: List of class names.
        """
        self.root_path = root_path
        self.observations = find_all_images_and_labels(root_path)
        self.transform = transform
        self.classes = sorted({x["label"] for x in self.observations})
        print(
            f"Found the following classes: {self.classes}, in total {len(self.observations)} images"
        )

    def __len__(self):
        return len(self.observations)

    def _get_one_item(self, idx: int):
        image_path = self.observations[idx]["image_path"]
        image = Image.open(image_path)
        label = self.observations[idx]["label"]
        label_num = self.classes.index(label)

        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label_num}

    def __getitem__(self, idx: int):
        return self._get_one_item(idx)

    def __getitems__(self, idx_list):
        return [self[idx] for idx in idx_list]

    @classmethod
    def from_subset(
        cls,
        original_dataset,
        subset_indices: list[int],
        transform: Callable | None = None,
    ):
        """
        Create a subset of the original dataset with only the specified indices.

        Args:
            original_dataset (ImageFolder): An instance of the ImageFolder dataset.
            subset_indices (List[int]): List of indices to create a subset of observations.
            transform: Override transform of current ds

        Returns:
            ImageFolder: A new instance of ImageFolder with the subset observations.
        """
        # Create a new instance with the same properties as the original
        subset_instance = cls(
            root_path=original_dataset.root_path,
            transform=original_dataset.transform if transform is None else transform,
        )

        # Filter the observations based on the subset indices
        subset_instance.observations = [original_dataset.observations[i] for i in subset_indices]
        subset_instance.classes = original_dataset.classes  # Keep class list consistent

        print(
            f"Created a subset with {len(subset_instance.observations)} images "
            f"from the original dataset of {len(original_dataset.observations)} images"
        )

        return subset_instance

class ImageFolderRandom(ImageFolder):
    """Modify parent class to return random image."""

    def __getitem__(self, idx: int):
        image_path = self.observations[idx]["image_path"]
        image = Image.open(image_path)
        label = self.observations[idx]["label"]
        label_num = self.classes.index(label)

        random_image = Image.fromarray(np.random.randint(0, 256, image.size, dtype=np.uint8))

        if self.transform:
            random_image = self.transform(random_image)
        return {"image": random_image, "label": label_num}
