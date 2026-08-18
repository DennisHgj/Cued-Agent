"""Backward-compatible import for the reorganized inference pipeline.

New code should import ``CuedAgentInference`` from ``cued_agent`` and use
``run_inference.py`` as the command-line entry point.
"""

from cued_agent.pipeline import CuedAgentInference

__all__ = ["CuedAgentInference"]


if __name__ == "__main__":
    from run_inference import main

    raise SystemExit(main())
