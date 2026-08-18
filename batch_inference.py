"""Batch Cued-Agent inference with one shared model instance."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cued-Agent on a video directory")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-ext", default=".mp4")
    parser.add_argument("--device")
    parser.add_argument("--hand-weight", type=float, default=4.5)
    parser.add_argument("--ctc-weight", type=float, default=0.5)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--lip-only", action="store_true")
    parser.add_argument("--no-self-correction", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_dir = args.video_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not video_dir.is_dir():
        print(f"Video directory not found: {video_dir}", file=sys.stderr)
        return 1

    videos = sorted(video_dir.glob(f"*{args.video_ext}"))
    if not videos:
        print(f"No {args.video_ext} videos found in {video_dir}", file=sys.stderr)
        return 1

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
        )
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video.name}")
        try:
            result = pipeline(video)
            result_path = output_dir / f"{video.stem}.json"
            with result_path.open("w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
            records.append(
                {"video": video.name, "status": "success", "output": str(result_path)}
            )
        except Exception as exc:
            records.append({"video": video.name, "status": "failed", "error": str(exc)})

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(records),
        "succeeded": sum(record["status"] == "success" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "records": records,
    }
    with (output_dir / "batch_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"Batch summary: {output_dir / 'batch_summary.json'}")
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
