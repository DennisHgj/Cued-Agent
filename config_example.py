"""Small programmatic configuration example.

Hydra owns the complete model architecture configuration. This helper composes
that configuration and avoids maintaining a second, incomplete Python schema.
"""

from __future__ import annotations

from pathlib import Path

from cued_agent.config import compose_inference_config


def make_inference_config(checkpoint: str | Path):
    return compose_inference_config(checkpoint)


if __name__ == "__main__":
    raise SystemExit(
        "Use: python run_inference.py --video VIDEO --checkpoint CHECKPOINT"
    )
