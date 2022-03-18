#!/usr/bin/env python

"""Model training entry point."""

import os
import shutil
import resource

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from core.logger import logger
from core.model import ImageClassifier
from core.utils import save_model, set_seed

FIRST_STAGE_DEBUG_EPOCHS = 2
SECOND_STAGE_DEBUG_EPOCHS = 2


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
    classifier = ImageClassifier(cfg=cfg)

    callbacks = classifier.get_callbacks()
    loggers = classifier.get_loggers()

    # Stage 1 -- backbone frozen
    logger.info("Stage 1 of the training -- backbone frozen...")

    # Freeze backbone parameters
    for param in classifier.feature_extractor.parameters():
        param.requires_grad = False

    # Make sure that parameters of FC layers are not frozen
    for param in classifier.classifier.parameters():
        param.requires_grad = True

    epochs = (
        FIRST_STAGE_DEBUG_EPOCHS
        if cfg.training.debug
        else cfg.training_first_stage_epochs
    )

    # Set logging steps to as fewer if possible if debugging
    if cfg.training.debug:
        cfg.trainer.log_every_n_steps = 1

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=epochs,
        **cfg.trainer,
    )
    trainer.fit(classifier)

    # Rename best checkpoint for the stage 1
    checkpoints_callback = [
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    ][0]
    best_checkpoint_path = checkpoints_callback.best_model_path
    logger.debug(f"Best checkpoint: {best_checkpoint_path}")

    stage1_checkpoint_path = os.path.join(
        cfg.logging.checkpoints_path, cfg.training.first_stage.best_model_name
    )
    logger.debug(f"Renaming to: {stage1_checkpoint_path}")
    shutil.copyfile(best_checkpoint_path, stage1_checkpoint_path)

    return classifier


def train_second_stage(cfg: DictConfig, classifier: ImageClassifier) -> None:
    """
    Run the second stage of the model training.

    All layers will be trained during this stage, including previously frozen
    backbone.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    classifier : ImageClassifier
        Pytorch Lightning Module object
    """
    loggers = classifier.get_loggers()
    callbacks = classifier.get_callbacks()

    # Stage 2 -- all layers unfrozen
    logger.info("Stage 2 of the training from the best checkpoint.")

    # Unfreeze backbone parameters
    for param in classifier.feature_extractor.parameters():
        param.requires_grad = True

    # Decrease learning rate because by the second stage we already have
    # decent convergence
    cfg.optimizer.params.lr = cfg.training.second_stage.learning_rate

    max_epochs = (
        FIRST_STAGE_DEBUG_EPOCHS + SECOND_STAGE_DEBUG_EPOCHS
        if cfg.training.debug
        else cfg.training.first_stage.epochs + cfg.training.second_stage.epochs
    )

    best_checkpoint_path = os.path.join(
        cfg.logging.checkpoints_path, cfg.training.first_stage.best_model_name
    )

    # Set logging steps to as fewer if possible if debugging
    if cfg.training.debug:
        cfg.trainer.log_every_n_steps = 1

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=max_epochs,
        **cfg.trainer,
    )

    trainer.fit(classifier, ckpt_path=best_checkpoint_path)

    # Rename best checkpoint for the stage 1
    checkpoints_callback = [
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    ][0]
    best_checkpoint_path = checkpoints_callback.best_model_path
    logger.debug(f"Best checkpoint: {best_checkpoint_path}")

    stage2_checkpoint_path = os.path.join(
        cfg.logging.checkpoints_path, cfg.training.second_stage.best_model_name
    )
    logger.debug(f"Renaming to: {stage2_checkpoint_path}")
    shutil.copyfile(best_checkpoint_path, stage2_checkpoint_path)


def save_best_model(cfg: DictConfig) -> None:
    """
    Save model from the best checkpoint.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration object
    """
    logger.info("Saving model from the best checkpoint...")

    best_model_path = os.path.join(
        cfg.logging.checkpoints_path, cfg.training.second_stage.best_model_name
    )
    model = ImageClassifier.load_from_checkpoint(best_model_path, cfg=cfg)
    save_model(model, cfg)

    logger.info("Done!")


@hydra.main(config_path="./", config_name="config")
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
