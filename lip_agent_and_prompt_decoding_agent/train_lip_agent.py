"""Fine-tune the lip encoder, CTC head, and attention decoder on lip ROI data."""

from __future__ import annotations

import inspect
import os

import hydra
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from datamodule.data_module_CCS import DataModule_CCS
from lightning_CCS import ModelModule_CCS
from training_contract import validate_training_contract


def _trainer_kwargs(trainer_cfg):
    """Translate trainer defaults across PyTorch Lightning 1.5 and 2.x."""
    kwargs = OmegaConf.to_container(trainer_cfg, resolve=True)
    parameters = inspect.signature(Trainer.__init__).parameters
    is_legacy_lightning = "use_distributed_sampler" not in parameters
    if (
        "use_distributed_sampler" in kwargs
        and is_legacy_lightning
        and "replace_sampler_ddp" in parameters
    ):
        kwargs["replace_sampler_ddp"] = kwargs.pop("use_distributed_sampler")
    if is_legacy_lightning:
        if kwargs.get("strategy") == "auto":
            kwargs["strategy"] = None
        if kwargs.get("accelerator") == "auto":
            kwargs["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
        if kwargs.get("devices") == "auto":
            kwargs["devices"] = 1
    unsupported = sorted(set(kwargs).difference(parameters))
    if unsupported:
        raise ValueError(
            "Unsupported Trainer options for this PyTorch Lightning version: "
            + ", ".join(unsupported)
        )
    return kwargs


@hydra.main(version_base="1.3", config_path="configs", config_name="config_CCS_lip_train")
def main(cfg):
    seed_everything(42, workers=True)
    if not cfg.data.dataset.root_dir:
        raise ValueError("Set data.dataset.root_dir to the local dataset root")
    initialization = validate_training_contract(cfg)
    print(f"Training initialization contract: {initialization}")
    experiment_dir = os.path.join(cfg.exp_dir, cfg.exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoint = ModelCheckpoint(
        monitor="loss_val",
        mode="min",
        dirpath=experiment_dir,
        save_last=True,
        filename="{epoch:02d}-{loss_val:.4f}",
        save_top_k=3,
    )
    trainer = Trainer(
        **_trainer_kwargs(cfg.trainer),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
    )
    modelmodule = ModelModule_CCS(cfg)
    datamodule = DataModule_CCS(cfg)
    trainer.fit(
        model=modelmodule,
        datamodule=datamodule,
        ckpt_path=cfg.resume_from_checkpoint,
    )
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
