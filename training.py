#!/usr/bin/env python

"""Model training entry point."""

import hydra
import os
import pytorch_lightning as pl
import resource

from omegaconf import DictConfig

from core.logger import get_logger
from core.model import ImageClassifier
from core.utils import flatten_omegaconf, save_best, set_seed


def train_first_stage(cfg: DictConfig) -> ImageClassifier:
    """
    Run the first stage of the model training.

    Backbone weights will be frozen during this stage and only fully-connected
    layers will be trained.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    hparams = flatten_omegaconf(cfg)
    classifier = ImageClassifier(hparams=hparams, cfg=cfg)

    callbacks = classifier.get_callbacks()
    loggers = classifier.get_loggers()

    # Stage 1 -- backbone frozen
    get_logger().info("Stage 1 of the training -- backbone frozen...")

    # Freeze backbone parameters
    for param in classifier.feature_extractor.parameters():
        param.requires_grad = False

    # Make sure that parameters of FC layers are not frozen
    for param in classifier.classifier.parameters():
        param.requires_grad = True

    trainer = pl.Trainer(
        logger=loggers,
        early_stop_callback=callbacks["early_stopping"],
        checkpoint_callback=callbacks["model_checkpoint"],
        max_epochs=cfg.training.first_stage.epochs,
        **cfg.trainer,
    )
    trainer.fit(classifier)

    return classifier


def train_second_stage(cfg: DictConfig, classifier) -> None:
    """
    Run the second stage of the model training.

    All layers will be trained during this stage, including previously frozen
    backbone.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    checkpoints = [
        ckpt
        for ckpt in os.listdir("./")
        if ckpt.endswith(".ckpt") and ckpt != "last.ckpt"
    ]

    best_checkpoint_path = checkpoints[0]
    stage1_checkpoint_name = cfg.training.first_stage.best_model_name
    os.rename(best_checkpoint_path, stage1_checkpoint_name)

    callbacks = classifier.get_callbacks()
    loggers = classifier.get_loggers()

    # Stage 2 -- all layers unfrozen
    get_logger().info("Stage 2 of the training from the best checkpoint.")

    # Unfreeze backbone parameters
    for param in classifier.feature_extractor.parameters():
        param.requires_grad = True

    # Decrease learning rate because by the second stage we already have
    # decent convergence
    cfg.optimizer.params.lr = cfg.training.second_stage.learning_rate

    max_epochs = cfg.training.first_stage.epochs + cfg.training.second_stage.epochs

    trainer = pl.Trainer(
        logger=loggers,
        early_stop_callback=callbacks["early_stopping"],
        checkpoint_callback=callbacks["model_checkpoint"],
        resume_from_checkpoint=stage1_checkpoint_name,
        max_epochs=max_epochs,
        **cfg.trainer,
    )

    trainer.fit(classifier)


def save_best_model(cfg: DictConfig) -> None:
    """
    Save model from the best checkpoint.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    hparams = flatten_omegaconf(cfg)
    get_logger().info("Saving model from the best checkpoint...")
    checkpoints = [
        ckpt
        for ckpt in os.listdir("./")
        if ckpt.endswith(".ckpt")
        and ckpt != "last.ckpt"
        and cfg.training.first_stage.best_model_name not in ckpt
    ]
    best_checkpoint_path = checkpoints[0]

    model = ImageClassifier.load_from_checkpoint(
        best_checkpoint_path, hparams=hparams, cfg=cfg
    )

    save_best(model, cfg)


@hydra.main(config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    """
    Run model training.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    set_seed(cfg.training.seed)

    # This is to avoid an issue: "RuntimeError: received 0 items of ancdata"
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    classifier = train_first_stage(cfg)
    train_second_stage(cfg, classifier)
    save_best_model(cfg)


if __name__ == "__main__":
    train()
