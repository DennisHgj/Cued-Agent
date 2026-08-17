"""Command-line entry point for one-video Cued-Agent inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the four-stage Cued-Agent inference pipeline"
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Fine-tuned lip encoder + CTC/attention decoder checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("inference_result.json")
    )
    parser.add_argument("--device", help="PyTorch device, for example cuda:0 or cpu")
    parser.add_argument("--hand-weight", type=float, default=4.5)
    parser.add_argument("--ctc-weight", type=float, default=0.5)
    parser.add_argument("--beam-size", type=int, default=20)

    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument(
        "--lip-only",
        action="store_true",
        help="Disable hand recognition and decode from lip features only",
    )
    hand_group.add_argument(
        "--hand-results",
        type=Path,
        help="Use precomputed hand-recognition JSON instead of calling OpenAI",
    )
    parser.add_argument(
        "--no-self-correction",
        action="store_true",
        help="Stop after phoneme decoding and do not call DeepSeek",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from cued_agent.config import compose_inference_config
        from cued_agent.pipeline import CuedAgentInference

        cfg = compose_inference_config(args.checkpoint)
        pipeline = CuedAgentInference(
            cfg,
            checkpoint_path=args.checkpoint,
            hand_weight=args.hand_weight,
            ctc_weight=args.ctc_weight,
            beam_size=args.beam_size,
            device=args.device,
            use_hand=not args.lip_only,
            use_self_correction=not args.no_self_correction,
            hand_results_path=args.hand_results,
        )
        result = pipeline(args.video)

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)

        print(f"Raw phonemes: {result['Raw_Cued_Speech_Sequence']}")
        print(f"Corrected phonemes: {result['Processed_Cued_Speech_Sequence']}")
        print(f"Mandarin: {result['Mandarin_Sequence']}")
        print(f"Saved: {output}")
        return 0
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
