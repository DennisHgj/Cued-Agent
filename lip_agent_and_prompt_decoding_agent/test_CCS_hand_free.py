"""Evaluate parameter-free hand-prompt decoding from a trained lip checkpoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from pytorch_lightning import Trainer

from datamodule.data_module_CCS import DataModule_CCS
from lightning_CCS_hand_prompt_decoding import ModelModule_CCS_Hand_prompt_decoding


@hydra.main(version_base="1.3", config_path="configs", config_name="config_CCS_hand_test")
def main(cfg):
    if not cfg.data.dataset.root_dir:
        raise ValueError("Set data.dataset.root_dir to the local dataset root")
    if not cfg.ckpt_path:
        raise ValueError("Set ckpt_path to a trained lip/decoder Lightning checkpoint")
    checkpoint = Path(str(cfg.ckpt_path)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(cfg.exp_dir) / cfg.exp_name / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    modelmodule = ModelModule_CCS_Hand_prompt_decoding(
        cfg, hand_weight=4.5, ctc_weight=0.5
    )
    datamodule = DataModule_CCS(cfg)
    trainer = Trainer(accelerator="auto", devices=1)
    trainer.test(model=modelmodule, datamodule=datamodule, ckpt_path=str(checkpoint))


if __name__ == "__main__":
    main()
