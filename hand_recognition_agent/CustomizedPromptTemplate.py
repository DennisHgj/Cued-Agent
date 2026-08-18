"""Multimodal hand position/shape recognition for Mandarin Cued Speech."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel

try:
    from .supportprompt_position import build_position_support_set
    from .supportprompt_shape import build_shape_support_set
except ImportError:  # Allow direct execution from this directory.
    from supportprompt_position import build_position_support_set
    from supportprompt_shape import build_shape_support_set


class Frame(BaseModel):
    frame_id: int
    hand_position: int
    hand_shape: int
    reasoning_process: str


class HandRecognition(BaseModel):
    # ``typing.List`` keeps Pydantic's runtime annotation evaluation compatible
    # with the original Python 3.8 ``auto_avsr`` research environment.
    recog_results: List[Frame]


BACKGROUND_PROMPT = """You are a Mandarin Cued Speech specialist. Classify the
right-hand position and shape in every test keyframe. Hand position labels are:
0 eye, 1 right side of head, 2 cheek, 3 chin, 4 below head. Hand shape labels
are: 0 index only; 1 index+middle together; 2 middle+ring+pinky; 3 all except
thumb; 4 all fingers; 5 thumb+index; 6 thumb+index+middle; 7 index+middle apart.
Use the visual support set, compare easily confused classes, and return one
result for every test frame in the same order.
"""


def _text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def _image(encoded_image: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}",
            "detail": "low",
        },
    }


def generate_recognition_single(
    hand_frames: list[str],
    seed: int = 3702,
    support_set_path: str = "",
    *,
    api_key: str | None = None,
    model: str | None = None,
    client: Any | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Recognize all keyframes using image content items and structured output."""
    if not hand_frames:
        return [], 0
    support_path = Path(support_set_path).expanduser().resolve()
    if not support_path.is_dir():
        raise FileNotFoundError(f"Hand support set not found: {support_path}")

    if client is None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for hand recognition. "
                "Use --lip-only or --hand-results to avoid the API call."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The openai Python package is required") from exc
        client = OpenAI(api_key=key)

    content: list[dict[str, Any]] = [_text(BACKGROUND_PROMPT)]
    content.extend(build_position_support_set(str(support_path)))
    content.extend(build_shape_support_set(str(support_path)))
    content.append(_text("Now classify these test keyframes in order:"))
    for frame_id, encoded_frame in enumerate(hand_frames):
        content.append(_text(f"Test keyframe {frame_id}"))
        content.append(_image(encoded_frame))

    completion = client.chat.completions.parse(
        model=model or os.getenv("OPENAI_HAND_MODEL", "gpt-4o-2024-08-06"),
        seed=seed,
        messages=[
            {
                "role": "system",
                "content": "Return structured hand-position and hand-shape labels.",
            },
            {"role": "user", "content": content},
        ],
        response_format=HandRecognition,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = getattr(completion.choices[0].message, "refusal", None)
        raise RuntimeError(f"Hand recognition returned no structured result: {refusal}")

    results = [item.model_dump() for item in parsed.recog_results]
    if len(results) != len(hand_frames):
        raise ValueError(
            "Hand recognition result count does not match keyframe count: "
            f"{len(results)} != {len(hand_frames)}"
        )
    usage = getattr(completion, "usage", None)
    return results, int(getattr(usage, "total_tokens", 0) or 0)


def process_video(video_path: str) -> list[str]:
    """Read a video and return base64 JPEG frames (legacy helper)."""
    import cv2

    video = cv2.VideoCapture(video_path)
    frames: list[str] = []
    try:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                frames.append(base64.b64encode(buffer.tobytes()).decode("ascii"))
    finally:
        video.release()
    return frames
