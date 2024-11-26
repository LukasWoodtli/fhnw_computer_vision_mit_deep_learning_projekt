"""Mostly taken (and adjusted) from:
https://github.com/marco-willi/cas-dl-compvis-exercises-hs2024
"""
import os
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

import json

from matplotlib import pyplot as plt
from robust_downloader import download
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import lightning as L

SCRIPT_DIR = Path(__file__).parent
label_names = json.load(open(SCRIPT_DIR / "part_names.json", "r"))
label_name_to_number = dict((reversed(item) for item in label_names.items()))


def in_colab():
    try:
        import google.colab

        in_colab = True
    except:
        in_colab = False

    print(f"In colab: {in_colab}")
    return in_colab


def setup_colab():
    from google.colab import drive
    drive.mount("/content/drive")


def download_and_unpack(url, filename, data_dir=SCRIPT_DIR / "data"):
    data_download_dir = Path(data_dir) / "download"
    data_download_dir.mkdir(parents=True, exist_ok=True)
    data_archive = data_download_dir / filename
    if not data_archive.exists():
        download(url, data_download_dir, filename)
    data_set_dir = data_dir / "data_set"
    if not data_set_dir.exists():
        data_set_dir.mkdir(parents=True, exist_ok=True)
        print(data_archive)
        with zipfile.ZipFile(data_archive, "r") as zipf:
            zipf.extractall(data_set_dir)
    return data_set_dir


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


def find_all_images_and_labels(image_dirs) -> list[dict]:
    """
    Load image paths and corresponding labels.

    Args:
        image_dirs: Directory (or list of directories) with all the images.

    Returns:
        A list of dicts, one for each obsevation
    """

    if not isinstance(image_dirs, list):
        image_dirs = [image_dirs]

    observations = []
    for image_dir in image_dirs:
        observations.extend(_find_all_images_and_labels_in_dir(image_dir))
    return observations



def _find_all_images_and_labels_in_dir(image_dir):
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

class DataSetModule(L.LightningDataModule):
    """Create a data module to manage train, validation and test sets."""

    def __init__(
        self,
        ds_train: Dataset,
        ds_val: Dataset,
        ds_test: Dataset,
        classes: list[str],
        train_transform: Callable | None,
        test_transform: Callable | None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        """Split the dataset into train, validation, and test sets."""
        if stage == "fit" or stage is None:
            if self.train_transform is not None:
                self.ds_train.transform = self.train_transform
            if self.test_transform is not None:
                self.ds_val.transform = self.test_transform

        if stage == "test" or stage is None:
            if self.test_transform is not None:
                self.ds_test.transform = self.test_transform

    def train_dataloader(self):
        """Return the train data loader."""
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Return the validation data loader."""
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return the test data loader."""
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

def create_train_test_split(
    ids: list[int],
    labels: list[int | str],
    random_state: int = 123,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:

    train_ids, test_ids = train_test_split(
        ids,
        stratify=labels,
        test_size=test_size,
        random_state=random_state,
    )

    train_ids, val_ids = train_test_split(
        train_ids,
        stratify=[labels[i] for i in train_ids],
        test_size=val_size,
        random_state=random_state,
    )

    return train_ids, val_ids, test_ids

def labels_hist(observations):
    l = [l["label"] for l in observations]
    plt.hist(l)
    plt.xticks(rotation=90)
    if len(l) > 15:
        plt.gca().set_xticklabels([])
    plt.show()

def plot_random_image_grid(data):
    # plot a grid
    cols, rows = 3, 3
    sample_ids = torch.randint(len(data), size=(cols * rows,))
    figure = plt.figure(figsize=(8, 8))
    for i in range(0, cols * rows):
        sample_idx = sample_ids[i]
        observation = data.observations[sample_idx]
        img_path = observation["image_path"]
        label = observation["label"]
        figure.add_subplot(rows, cols, i + 1)
        plt.title(label[:20])
        plt.axis("off")
        img = plt.imread(img_path)
        plt.imshow(img, cmap="gray")
    plt.show()

    
def print_tensorboard_cmd():
    logdir = SCRIPT_DIR / "lightning_logs"
    print("Run in venv:")
    print(" ".join(['tensorboard', f'--logdir="{logdir}"', '--host=127.0.0.1', '--port=6006']))
    print(f"Log dir: {logdir}\nserver on: http://127.0.0.1:6006")

