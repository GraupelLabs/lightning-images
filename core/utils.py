"""Module with utility functions."""

from collections import OrderedDict
from typing import Any, Tuple, Dict, List

import importlib
import json
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch

from hydra.utils import to_absolute_path
from shutil import copyfile
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

IMPORT_SHORTCUTS = {"A": "albumentations"}


def collate_fn(batch):
    """TODO Add missing docstring."""
    return tuple(zip(*batch))


def flatten_omegaconf(d, sep="_"):
    """TODO Add missing docstring."""
    d = OmegaConf.to_container(d)
    obj = OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float]}

    return obj


def load_augmentations(augmentations: DictConfig) -> Any:
    """Load augmentation transformations from config.

    Parameters
    ----------
    cfg : DictConfig
        Augmentation transforms dictionary

    Returns
    -------
    Any
        Augmentations composition
    """
    transform_name = augmentations["transform"]
    has_nested_transforms = "transforms" in augmentations

    if has_nested_transforms:
        transforms = [
            load_augmentations(transform) for transform in augmentations["transforms"]
        ]

        return load_obj(transform_name)(transforms)
    else:
        parameters = {}

        for transform_key in augmentations:
            if transform_key != "transform":
                parameters[transform_key] = augmentations[transform_key]

        return load_obj(transform_name)(**parameters)


# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py # noqa
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.

    Parameters
    ----------
    obj_path : str
        Path to an object to be extracted, including the object name.
    default_obj_path : str, optional
        Default object path., by default ""

    Returns
    -------
    Any
        Extracted object.

    Raises
    ------
    AttributeError
        When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.split(".")[:-1]
    obj_name = obj_path.rsplit(".", 1)[-1]

    # Transform module path so shortcuts will be converted to full module names
    obj_path = ".".join(
        [IMPORT_SHORTCUTS[x] if x in IMPORT_SHORTCUTS else x for x in obj_path_list]
    )

    if obj_path == "":
        obj_path = default_obj_path

    # Check if the module is already imported, otherwise try to import it
    if obj_path not in sys.modules:
        module_obj = importlib.import_module(obj_path)
    else:
        module_obj = sys.modules[obj_path]

    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")

    return getattr(module_obj, obj_name)


def save_model(model: pl.LightningModule, cfg: DictConfig) -> None:
    """
    Save the entire model.

    Parameters
    ----------
    model : pl.LightningModule
        Pytorch Lightning model
    cfg : DictConfig
        Project config with logs path
    """
    best_model_folder = os.path.dirname(cfg.logging.best_model_path)
    best_model_path = os.path.join(best_model_folder, cfg.logging.best_model_name)
    labels_path = os.path.join(best_model_folder, cfg.logging.best_model_labels)

    if not os.path.exists(best_model_folder):
        os.makedirs(best_model_folder)

    torch.save(model, best_model_path)

    labels_origin_path = to_absolute_path(cfg.data.labels_file_path)
    copyfile(labels_origin_path, labels_path)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed globally.

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)


def create_class_mapping(
    mapping_path: str,
) -> Tuple[Dict[int, str], List[str]]:
    """Read the class mapping from file and swap keys with values.

    Parameters
    ----------
    mapping_path : str
        Path to the mapping file.

    Returns
    -------
    Tuple[Dict[int, str], List[str]]
        - Classes map, e.g.
            {
                0: "ice",
                1: "drop",
                2: "plate",
                ...
            }
        - List of class names, such as
            ["ice", "drop", "plate", ...]

    Raises
    ------
    Exception
        Could not read the class mapping from file
    """
    class_map = None
    classes = []

    with open(mapping_path, "r") as json_file:
        class2label = json.load(json_file)
        classes = list(class2label.keys())
        class_map = {v: k for k, v in class2label.items()}

    if not class_map or len(class_map) == 0:
        raise Exception("Incorrect label-->class mapping.")

    return class_map, classes
