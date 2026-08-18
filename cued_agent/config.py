"""Hydra configuration helpers used by the command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_DIR = PROJECT_ROOT / "lip_agent_and_prompt_decoding_agent" / "configs"


def compose_inference_config(checkpoint_path: str | Path) -> Any:
    """Compose the model architecture config and attach a fine-tuned checkpoint."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fine-tuned lip/decoder checkpoint not found: {checkpoint}")

    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError(
            "hydra-core is required to compose the model configuration. "
            "Install the inference dependencies first."
        ) from exc

    with initialize_config_dir(
        version_base="1.3", config_dir=str(MODEL_CONFIG_DIR)
    ):
        cfg = compose(config_name="config_CCS_hand_infer")

    # A fine-tuned Lightning checkpoint contains the lip encoder, CTC projection,
    # and attention decoder. The base pretraining path must stay empty here so the
    # model is initialized only once before loading the fine-tuned state.
    cfg.ckpt_path = str(checkpoint)
    cfg.pretrained_model_path = None
    return cfg
