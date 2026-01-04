# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

import random
import numpy as np
from torch.utils.data import Dataset
import pathlib
import cv2
import yaml
from albumentations import Compose
from hydra.utils import get_original_cwd


class DET_DS(Dataset):
    """
    A custom dataset class for object detection tasks.

    This dataset class loads images and their corresponding labels from specified directories,
    applies transformations if provided, and returns the processed image along with target annotations.

    Attributes:
        root_dir (str): The root directory containing the dataset.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        config_file (str): The path to the configuration file containing class names and folder structure.
        transforms (Albumentations Compose, optional): A function or object to apply transformations to the images and annotations.
        image_paths (list): A list of valid image file paths.
        labels (list): A list of class names.
        id2lbl (dict): A mapping from class IDs to class names.
        lbl2id (dict): A mapping from class names to class IDs.

    The configuration file (YAML) should contain:
        - names: List of class names
        - folders (optional): Dictionary with keys 'images', 'labels', 'train', 'val', 'test'
          specifying the folder names. Defaults to standard names if not provided.
        - folders.structure (optional): Either 'type_first' (default) for images/train structure
          or 'split_first' for train/images structure.

    Methods:
        __len__(): Returns the number of valid images in the dataset.
        __getitem__(idx): Returns the processed image and target annotations for the given index.

    Args:
        root_dir (str): The root directory containing the dataset.
        split (str): The dataset split (e.g., 'train', 'val', 'test').
        config_file (str): The path to the configuration file containing class names and folder structure.
        transforms (Albumentations Compose, optional): A function or object to apply transformations to the images and annotations.

    Raises:
        FileNotFoundError: If the configuration file or label files are not found.
        ValueError: If an image cannot be loaded or is invalid.
    """

    def __init__(
        self,
        root_dir: str | None,
        split: str,
        config_file: str,
        transforms: Compose | None = None,
        processor=None,
        normalize: bool = False,
    ):
        self.root_dir: pathlib.Path
        if root_dir is None:
            self.root_dir = pathlib.Path(get_original_cwd()) / "data"
        else:
            self.root_dir = pathlib.Path(get_original_cwd()) / "data" / root_dir
        self.split = split
        self.transforms = transforms
        self.config_dir = self.root_dir / config_file
        self.processor = processor
        self.normalize = normalize

        # Load config to get folder structure
        with open(self.config_dir, "r") as f:
            config = yaml.safe_load(f)

        # Get folder configuration with defaults for backwards compatibility
        folders = config.get("folders", {})
        image_folder = folders.get("images", "images")
        label_folder = folders.get("labels", "labels")
        split_folder = folders.get(split, split)

        # Get structure type: "type_first" (images/train) or "split_first" (train/images)
        structure = folders.get("structure", "type_first")

        if structure == "split_first":
            # Structure: root/train/images, root/train/labels
            self.image_dir = self.root_dir / split_folder / image_folder
            self.label_dir = self.root_dir / split_folder / label_folder
        else:
            # Structure: root/images/train, root/labels/train (default)
            self.image_dir = self.root_dir / image_folder / split_folder
            self.label_dir = self.root_dir / label_folder / split_folder
        raw_paths = sorted(list(pathlib.Path(self.image_dir).glob("*.jpg")))

        self.labels = [name for name in config["names"]]
        self.id2lbl = dict(enumerate(self.labels))
        self.lbl2id = {v: k for k, v in self.id2lbl.items()}

        dropped_classes = {
            class_name: {"missing_label": 0, "corrupted_image": 0}
            for class_name in self.labels
        }
        total_drops = {"missing_label": 0, "corrupted_image": 0}

        num_dropped = 0
        valid_imgs = []
        valid_labels = []

        for p in raw_paths:
            label_path = pathlib.Path(self.label_dir) / (p.stem + ".txt")

            img = cv2.imread(str(p))
            img_valid = img is not None
            label_exists = label_path.exists()

            if img_valid and label_exists:
                valid_imgs.append(p)
                valid_labels.append(label_path)
            else:
                num_dropped += 1
                affected_classes = set()

                if label_exists:
                    try:
                        with open(label_path, "r") as f:
                            for line in f.readlines():
                                if line.strip():
                                    cls_id = int(float(line.strip().split()[0]))
                                    if cls_id in self.id2lbl:
                                        affected_classes.add(self.id2lbl[cls_id])
                    except (ValueError, IndexError):
                        print(f"[WARN] malformed label file {label_path.name}")

                if not img_valid:
                    total_drops["corrupted_image"] += 1
                    for class_name in affected_classes:
                        dropped_classes[class_name]["corrupted_image"] += 1
                    print(f"[WARN] dropping corrupted image {p.name}")

                elif not label_exists:
                    total_drops["missing_label"] += 1
                    print(f"[WARN] dropping image {p.name} due to missing label")

        print(f"\n=== Dataset Statistics for {split} ===")
        print(f"Total images processed: {len(raw_paths)}")
        print(f"Valid images: {len(valid_imgs)}")
        print(f"Dropped images: {num_dropped}")
        print(f"  - Corrupted images: {total_drops['corrupted_image']}")
        print(f"  - Missing labels: {total_drops['missing_label']}")

        print("\n=== Per-Class Drop Statistics ===")
        any_drops = False
        for class_name, drops in dropped_classes.items():
            total_class_drops = sum(drops.values())
            if total_class_drops > 0:
                any_drops = True
                print(f"{class_name}:")
                if drops["corrupted_image"] > 0:
                    print(f"  - Corrupted images: {drops['corrupted_image']}")
                if drops["missing_label"] > 0:
                    print(f"  - Missing labels: {drops['missing_label']}")
                print(f"  - Total drops: {total_class_drops}")

        if not any_drops:
            print(
                "No per-class drops to report (missing labels can't be tracked per class)"
            )

        self.image_paths = valid_imgs
        self.label_paths = valid_labels

    def __len__(self):
        """
        Returns:
            int: The number of valid images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the processed image and target annotations for the given index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing:
                - img (numpy.ndarray): The processed image.
                - target (dict): A dictionary containing target annotations, including:
                    - image_id (int): The index of the image.
                    - annotations (list): A list of dictionaries with bounding box, category ID, area, and iscrowd flag.
                    - orig_size (torch.Tensor): The original size of the image (height, width).
        """
        image_path = self.image_paths[idx]
        label_path = pathlib.Path(self.label_dir) / (image_path.stem + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            new_idx = random.randrange(len(self.image_paths))
            print("img is empty")
            return self.__getitem__(new_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        height, width = img.shape[:2]
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, cx, cy, w, h = map(float, line.strip().split())

                x1 = (cx - w / 2) * width
                y1 = (cy - h / 2) * height
                box_w = w * width
                box_h = h * height
                boxes.append([x1, y1, box_w, box_h])
                labels.append(int(cls))

        boxes, labels = (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )
        if self.transforms:
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        target = format_for_hf_processor(boxes, labels, idx)

        if hasattr(self, "processor") and self.processor:
            result = self.processor(
                images=img,
                annotations=target,
                return_tensors="pt",
            )
            result = {k: v[0] for k, v in result.items()}
            return result
        else:
            raise AttributeError("No Processor in dataset")

    def get_raw_item(self, idx: int):
        """
        Fetches a raw, untransformed image and its annotations.
        This is a helper method for multi-sample augmentations like Mosaic.
        """
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        img = cv2.imread(str(image_path))
        if img is None:  # Handle potential bad image
            return self.get_raw_item(np.random.randint(0, len(self)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        height, width, _ = img.shape

        with open(label_path, "r") as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = map(float, line.strip().split())
                x1 = (cx - w / 2) * width
                y1 = (cy - h / 2) * height
                box_w = w * width
                box_h = h * height
                boxes.append([x1, y1, box_w, box_h])
                labels.append(int(cls))

        return img, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def format_for_hf_processor(boxes, labels, idx):
    """Convert back to HF format"""
    return {
        "image_id": idx,
        "annotations": [
            {
                "bbox": box.tolist() if hasattr(box, "tolist") else box,
                "category_id": int(label),
                "area": float(box[2] * box[3]),
                "iscrowd": 0,
            }
            for box, label in zip(boxes, labels)
        ],
    }
