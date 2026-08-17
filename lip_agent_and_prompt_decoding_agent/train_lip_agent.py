"""Fine-tune the lip encoder, CTC head, and attention decoder on lip ROI data."""

from __future__ import annotations

import os

import hydra
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datamodule.data_module_CCS import DataModule_CCS
from lightning_CCS import ModelModule_CCS


@hydra.main(version_base="1.3", config_path="configs", config_name="config_CCS_lip_train")
def main(cfg):
    seed_everything(42, workers=True)
    if not cfg.data.dataset.root_dir:
        raise ValueError("Set data.dataset.root_dir to the local dataset root")
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
        **cfg.trainer,
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
