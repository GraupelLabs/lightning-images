#!/usr/bin/env python

"""Model evaluation entry point."""

import os

import hydra
import numpy as np
import pandas as pd
import resource
import torch

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from tqdm import tqdm

from core.dataset import get_training_dataset
from core.logger import get_logger
from core.utils import collate_fn, set_seed, create_class_mapping


def run_predictions(cfg, model, dataset):
    """TODO Add missing docstring."""
    data_loadder = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_true_labels = []
    all_predicted_labels = []
    all_probas = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for images, labels, _ in tqdm(data_loadder):
        images = [image.to(device) for image in images]
        images = torch.stack(images, dim=0)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        true_labels = torch.cat(labels).cpu().numpy()
        predicted_labels = probabilities.argmax(dim=1).cpu().numpy()

        all_true_labels.extend(true_labels.tolist())
        all_predicted_labels.extend(predicted_labels.tolist())
        all_probas.extend(probabilities.detach().cpu().numpy())

    class_mappings, class_names = create_class_mapping(
        os.path.join(
            to_absolute_path(cfg.logging.best_model_path),
            cfg.logging.best_model_labels,
        )
    )

    df_proba = pd.DataFrame(np.asarray(all_probas), columns=class_names)
    df_proba["label"] = np.asarray(all_predicted_labels).reshape(df_proba.shape[0], 1)
    df_proba.to_csv("valid.csv", index=False)

    report = classification_report(
        all_true_labels,
        all_predicted_labels,
        labels=list(class_mappings.keys()),
        target_names=class_names,
        digits=3,
    )

    return report


@hydra.main(config_path="./", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run model inference on the validation dataset.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    set_seed(cfg.training.seed)

    # This is to avoid an issue: "RuntimeError: received 0 items of ancdata"
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = os.path.join(
        to_absolute_path(cfg.logging.best_model_path),
        cfg.logging.best_model_name,
    )

    model = torch.load(model_path, map_location=device)
    model.eval()

    dataset = get_training_dataset(cfg)
    test_dataset = dataset["valid"]

    get_logger().info("Evaluating the test data...")
    classification_report = run_predictions(cfg, model, test_dataset)

    print(classification_report)
    print()


if __name__ == "__main__":
    main()
