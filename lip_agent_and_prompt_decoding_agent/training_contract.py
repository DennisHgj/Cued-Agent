"""Fail-fast checks for reproducible lip/decoder training runs."""

from __future__ import annotations

from pathlib import Path


def _configured_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(text).expanduser()


def validate_compatible_fraction(loaded, total, minimum):
    """Reject checkpoints that only happen to match a small part of the model."""
    minimum = float(minimum)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("min_pretrained_tensor_fraction must be between 0 and 1")
    if total <= 0:
        raise ValueError("Checkpoint target must contain at least one tensor")
    fraction = float(loaded) / float(total)
    if fraction < minimum:
        raise ValueError(
            "Pretrained checkpoint compatibility is below the required threshold: "
            f"{loaded}/{total} tensors ({fraction:.2%}) < {minimum:.2%}"
        )
    return fraction


def validate_training_contract(cfg):
    """Validate initialization and batching before allocating a model or dataset."""
    pretrained = _configured_path(getattr(cfg, "pretrained_model_path", None))
    resume = _configured_path(getattr(cfg, "resume_from_checkpoint", None))
    allow_random = bool(getattr(cfg, "allow_random_initialization", False))

    if pretrained is None and resume is None and not allow_random:
        raise ValueError(
            "Cued-Agent reproduction training requires pretrained_model_path or "
            "resume_from_checkpoint. Set allow_random_initialization=true only "
            "for an explicit from-scratch control run."
        )

    for label, path in (
        ("pretrained_model_path", pretrained),
        ("resume_from_checkpoint", resume),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    data_cfg = cfg.data
    if bool(getattr(data_cfg, "include_hand", False)):
        raise ValueError(
            "train_lip_agent.py trains only the lip encoder and decoder; "
            "data.include_hand must remain false"
        )
    for name in ("max_frames", "max_frames_val"):
        if int(getattr(data_cfg, name)) <= 0:
            raise ValueError(f"data.{name} must be positive")
    if int(getattr(data_cfg, "num_workers", 0)) < 0:
        raise ValueError("data.num_workers must be non-negative")

    trainer_cfg = cfg.trainer
    if int(getattr(trainer_cfg, "accumulate_grad_batches", 1)) <= 0:
        raise ValueError("trainer.accumulate_grad_batches must be positive")

    validate_compatible_fraction(
        1,
        1,
        getattr(cfg, "min_pretrained_tensor_fraction", 0.0),
    )
    if resume is not None:
        return "resume"
    if pretrained is not None:
        return "pretrained"
    return "random"
