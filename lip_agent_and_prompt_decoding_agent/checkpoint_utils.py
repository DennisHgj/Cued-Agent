"""Checkpoint loading shared by lip training and hand-prompt inference models."""

from __future__ import annotations

import torch


def _state_dict(payload):
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    for key in ("model_state_dict", "state_dict"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def _strip_prefix(state, prefix):
    return {
        (key[len(prefix):] if str(key).startswith(prefix) else str(key)): value
        for key, value in state.items()
    }


def load_compatible_weights(
    model,
    checkpoint_path,
    *,
    transfer_frontend=False,
    transfer_encoder=False,
):
    """Load every name-and-shape compatible tensor and report the transfer size."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = _strip_prefix(_state_dict(payload), "module.")

    if transfer_frontend:
        target = model.encoder.frontend
        for prefix in ("model.encoder.frontend.", "encoder.frontend.", "model."):
            state = _strip_prefix(state, prefix)
    elif transfer_encoder:
        target = model.encoder
        for prefix in ("model.encoder.", "encoder.", "model."):
            state = _strip_prefix(state, prefix)
    else:
        target = model
        state = _strip_prefix(state, "model.")

    target_state = target.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    if not compatible:
        raise ValueError(
            f"No compatible tensors found in checkpoint: {checkpoint_path}"
        )
    target.load_state_dict(compatible, strict=False)
    return len(compatible), len(target_state)
