"""Image classification dataset class."""

from typing import Dict, Tuple

import cv2
import json
import numpy as np
import os
import pandas as pd
import torch

from albumentations.core.composition import Compose
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from .logger import get_logger
from .utils import load_augmentations


class ImagesDataset(Dataset):
    """Images dataset class."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        cfg: DictConfig,
        transforms: Compose,
        mode: str = "train",
    ):
        """
        Prepare data for object detection on chest X-ray images.

        Parameters
        ----------
        dataframe : pd.DataFrame, optional
            dataframe with image paths and labels assigned to them
        mode : str, optional
            train/val/test, by default "train"
        cfg : DictConfig, optional
            config with parameters, by default None
        transforms : Compose, optional
            albumentations, by default None
        """
        self.df = dataframe
        self.mode = mode
        self.cfg = cfg
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor, str]:
        """
        Get dataset item.

        Parameters
        ----------
        idx : int
            Dataset item index

        Returns
        -------
        Tuple[Tensor, Dict[str, Tensor], str]
            (image, target, image_id)
        """
        data_entry = self.df.iloc[idx]
        image_path = data_entry["image"]
        image_id = os.path.basename(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)

        if self.mode == "test":
            label = torch.Tensor([0]).type("torch.LongTensor")
        else:
            label = torch.Tensor([data_entry["label"]]).type("torch.LongTensor")

        image = self.transforms(image=image)["image"]

        return image, label, image_id

    def __len__(self) -> int:
        """
        Get dataset size.

        Returns
        -------
        int
            Dataset size
        """
        return len(self.df)


def data_ready(cfg: DictConfig) -> bool:
    """
    Check if dataset was generated from images folder.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration

    Returns
    -------
    bool
        True if dataset was generated, otherwise False
    """
    data_path = to_absolute_path(cfg.data.dataset_path)
    images_folder_name = cfg.data.images_folder_name
    dataset_filename = cfg.data.dataset_filename
    labels_filename = cfg.data.labels_filename

    data_folder_entries = os.listdir(data_path)

    if (
        images_folder_name in data_folder_entries
        and dataset_filename in data_folder_entries
        and labels_filename in data_folder_entries
    ):
        return True

    return False


def get_training_dataset(cfg: DictConfig) -> Dict[str, Dataset]:
    """
    Get training and validation datasets.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration

    Returns
    -------
    Dict[str, Dataset]
        {"train": train_dataset, "valid": valid_dataset}
    """
    if not data_ready(cfg):
        prepare_dataset(cfg)

    data_path = to_absolute_path(cfg.data.dataset_path)
    dataset_path = os.path.join(data_path, cfg.data.dataset_filename)
    data = pd.read_csv(to_absolute_path(dataset_path))

    train_df, valid_df = train_test_split(
        data,
        test_size=cfg.data.validation_split,
        stratify=data.label,
        random_state=cfg.training.seed,
    )

    # for fast training
    if cfg.training.debug:
        train_df = train_df[:100]
        valid_df = valid_df[:100]

    train_augs = load_augmentations(cfg["augmentations"]["train"])
    valid_augs = load_augmentations(cfg["augmentations"]["valid"])

    train_dataset = ImagesDataset(train_df, cfg, train_augs, "train")
    valid_dataset = ImagesDataset(valid_df, cfg, valid_augs, "valid")

    return {"train": train_dataset, "valid": valid_dataset}


def prepare_dataset(cfg: DictConfig) -> None:
    """
    Generate dataset files from the images folder.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration

    Raises
    ------
    Exception
        Images folder not found
    """
    data_path = to_absolute_path(cfg.data.dataset_path)
    images_folder_name = cfg.data.images_folder_name
    images_folder_path = os.path.join(data_path, images_folder_name)
    dataset_filename = cfg.data.dataset_filename
    labels_filename = cfg.data.labels_filename
    num_files_total = sum(
        [len(files) for root, dirs, files in os.walk(images_folder_path)]
    )

    data_folder_entries = os.listdir(data_path)

    if images_folder_name not in data_folder_entries:
        raise Exception("Cannot generate a dataset, images folder not found.")

    image_paths = []
    image_classes = []

    get_logger().info("Generating dataset...")
    progress_bar = tqdm(total=num_files_total)

    for root, _, files in os.walk(images_folder_path):
        for cur_file_name in files:
            cur_dir_name = os.path.basename(root.replace(images_folder_path, ""))

            if cur_dir_name == "":
                continue

            cur_filepath = os.path.join(root, cur_file_name)
            image_paths.append(cur_filepath)
            image_classes.append(cur_dir_name)

            progress_bar.update(1)

    progress_bar.close()

    label_encoder = LabelEncoder()
    image_labels = label_encoder.fit_transform(image_classes)

    data_classes = label_encoder.classes_
    data_labels = label_encoder.transform(data_classes)
    # This is needed because JSON cannot serialize int64 objects
    data_labels = [int(x) for x in data_labels]

    # Save dataset into the csv file
    dataset = pd.DataFrame(
        np.column_stack([image_paths, image_labels]),
        columns=["image", "label"],
    )
    dataset_file_path = os.path.join(data_path, dataset_filename)
    dataset.to_csv(dataset_file_path, index=False)
    get_logger().info("Dataset file created: %s", dataset_file_path)

    # Save class names to labels mapping
    tag2label = dict(zip(data_classes, data_labels))
    tag2label_file_path = os.path.join(data_path, labels_filename)
    with open(tag2label_file_path, "w") as json_file:
        json.dump(tag2label, json_file, indent=4)
    get_logger().info("Labels file created: %s", tag2label_file_path)
