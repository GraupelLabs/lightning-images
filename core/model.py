"""Image classification model class."""

from typing import List

import cv2
import os
import torch

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchmetrics import Accuracy

from .dataset import get_training_dataset
from .logger import get_logger
from .utils import collate_fn, load_obj


class Identity(nn.Module):
    """Network identity (empty) layer."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Make a forward pass of the input through the model."""
        return x


class ImageClassifier(LightningModule):
    """Classification pytorch module."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.feature_extractor = load_obj(cfg.model.backbone.class_name)
        self.feature_extractor = self.feature_extractor(**cfg.model.backbone.params)
        self.feature_extractor.fc = Identity()

        fc_first_layer_size = 512
        fc_second_layer_size = 128

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(num_features=fc_first_layer_size),
            torch.nn.Linear(
                in_features=fc_first_layer_size,
                out_features=fc_second_layer_size,
                bias=False,
            ),
            torch.nn.BatchNorm1d(num_features=fc_second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(
                in_features=fc_second_layer_size,
                out_features=cfg.data.num_classes,
                bias=False,
            ),
        )

        self.criterion = load_obj(cfg.training.loss)()
        self.accuracy = Accuracy()

    def _save_images(self, images, batch_id: int = 0, mode: str = "train") -> None:
        """TODO Add missing docstring."""
        images_folder = os.path.join(self.cfg.debug.images_folder, mode)

        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        for img_id, img in enumerate(images):
            img_data = img.permute(2, 1, 0).cpu().detach().numpy()
            img_data = cv2.normalize(
                img_data,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
            image_name = f"{batch_id}_{img_id}.jpg"
            image_path = os.path.join(images_folder, image_name)
            cv2.imwrite(image_path, img_data)

    def configure_optimizers(self):
        """TODO Add missing docstring."""
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.parameters(), **self.cfg.optimizer.params
        )

        scheduler = load_obj(self.cfg.scheduler.class_name)(
            optimizer, **self.cfg.scheduler.params
        )

        return [optimizer], [scheduler]

    def forward(self, image, *args, **kwargs):
        """Pass input through the backbone into the Linear head."""
        representations = self.feature_extractor(image)

        output = self.classifier(representations)
        return output

    def get_callbacks(self) -> List[Callback]:
        """
        Get a list of pytorch callbacks for this model.

        Returns
        -------
        List[Callback]
            List of callbacks
        """
        callbacks = [
            load_obj(callback.class_name)(**callback.params)
            for callback in self.cfg.callbacks.values()
        ]
        return callbacks

    def get_loggers(self) -> List:
        """TODO Add missing docstring."""
        return [TensorBoardLogger(save_dir=self.cfg.logging.logs_dir)]

    def prepare_data(self):
        """TODO Add missing docstring."""
        get_logger().info("Loading training dataset...")
        datasets = get_training_dataset(self.cfg)
        self.train_dataset = datasets["train"]
        self.valid_dataset = datasets["valid"]

    def train_dataloader(self):
        """TODO Add missing docstring."""
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

        return train_loader

    def training_step(self, batch, batch_nb):
        """TODO Add missing docstring."""
        images, labels, _ = batch
        batch_size = len(images)
        images = torch.stack(images, dim=0)

        if self.cfg.debug.enabled and self.cfg.debug.save_images:
            self._save_images(images, batch_nb, "train")

        labels = torch.stack(labels, dim=0).squeeze()

        labels_predicted = self(images)

        loss = self.criterion(labels_predicted, labels)
        self.log("train_loss", loss, on_epoch=True, batch_size=batch_size)

        return loss

    def val_dataloader(self):
        """TODO Add missing docstring."""
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

        return valid_loader

    def validation_step(self, batch, batch_nb):
        """TODO Add missing docstring."""
        images, labels, _ = batch
        batch_size = len(images)
        images = torch.stack(images, dim=0)

        if self.cfg.debug.enabled and self.cfg.debug.save_images:
            self._save_images(images, batch_nb, "valid")

        labels = torch.stack(labels, dim=0).squeeze()

        labels_predicted = self(images)

        val_loss = self.criterion(labels_predicted, labels)
        self.accuracy(labels_predicted, labels)
        self.log(
            "val_loss", val_loss, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "val_acc",
            self.accuracy,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return val_loss
