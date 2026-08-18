"""Pure hand-keyframe and hand-prompt utilities.

The paper's hand prompt H has shape ``[T, q]``. In this implementation ``q=44``:
42 decoder entries (phonemes plus unknown and separator) and CTC blank/EOS make
44 columns. Hand labels only activate compatible phoneme columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


TOKEN_TO_ID = {
    "<blank>": 0,
    "<unk>": 1,
    "b": 2,
    "p": 3,
    "m": 4,
    "f": 5,
    "d": 6,
    "t": 7,
    "n": 8,
    "l": 9,
    "g": 10,
    "k": 11,
    "h": 12,
    "j": 13,
    "q": 14,
    "x": 15,
    "zh": 16,
    "ch": 17,
    "sh": 18,
    "r": 19,
    "z": 20,
    "c": 21,
    "s": 22,
    "y": 23,
    "w": 24,
    "yu": 25,
    "a": 26,
    "o": 27,
    "e": 28,
    "i": 29,
    "u": 30,
    "v": 31,
    "ai": 32,
    "ei": 33,
    "ao": 34,
    "ou": 35,
    "er": 36,
    "an": 37,
    "en": 38,
    "ang": 39,
    "eng": 40,
    "ong": 41,
    "-": 42,
}
VOCAB_SIZE = 44

POSITION_PHONEMES = {
    0: ("an", "e", "o"),
    1: ("a", "ou", "er", "en"),
    2: ("i", "v", "ang"),
    3: ("ai", "u", "ao"),
    4: ("eng", "ong", "ei"),
}

SHAPE_PHONEMES = {
    0: ("p", "d", "zh"),
    1: ("k", "q", "z"),
    2: ("s", "r", "h"),
    3: ("b", "n", "yu"),
    4: ("m", "t", "f"),
    5: ("l", "x", "w"),
    6: ("g", "j", "ch"),
    7: ("y", "c", "sh"),
}


def find_slow_motion_groups(
    hand_positions: np.ndarray | Sequence[Sequence[float]],
    movement_threshold: float = 6.0,
    index_gap_threshold: int = 2,
) -> list[list[int]]:
    """Return compact-frame index groups that satisfy the paper's slow-motion rule."""
    positions = np.asarray(hand_positions, dtype=np.float32)
    if positions.size == 0:
        return []
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"hand_positions must have shape [N, 2], got {positions.shape}"
        )
    if movement_threshold < 0:
        raise ValueError("movement_threshold must be non-negative")
    if index_gap_threshold < 1:
        raise ValueError("index_gap_threshold must be at least 1")

    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    slow_indices = (np.flatnonzero(distances <= movement_threshold) + 1).tolist()
    if not slow_indices:
        return []

    groups: list[list[int]] = [[slow_indices[0]]]
    for index in slow_indices[1:]:
        if index - groups[-1][-1] <= index_gap_threshold:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def select_keyframes(groups: Sequence[Sequence[int]]) -> list[int]:
    """Select the middle compact-frame index from every slow-motion group."""
    keyframes: list[int] = []
    for group in groups:
        if not group:
            raise ValueError("slow-motion groups must not contain empty groups")
        keyframes.append(int(group[(len(group) - 1) // 2]))
    return keyframes


def _field(result: Any, *names: str) -> Any:
    for name in names:
        if isinstance(result, Mapping) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    raise KeyError(f"Recognition result is missing one of: {', '.join(names)}")


def _labels(result: Any) -> tuple[int, int]:
    position = int(_field(result, "hand_position"))
    shape = int(_field(result, "hand_shape", "hand_gesture"))
    if position not in POSITION_PHONEMES:
        raise ValueError(f"hand_position must be in [0, 4], got {position}")
    if shape not in SHAPE_PHONEMES:
        raise ValueError(f"hand_shape must be in [0, 7], got {shape}")
    return position, shape


def build_hand_prompt(
    recognition_results: Sequence[Any],
    slow_motion_groups: Sequence[Sequence[int]],
    valid_frame_indices: Sequence[int] | np.ndarray,
    total_frames: int,
    vocab_size: int = VOCAB_SIZE,
) -> np.ndarray:
    """Build a frame-aligned hand prompt matrix from compact detected-hand frames.

    ``valid_frame_indices`` maps every compact hand frame back to its zero-based
    index in the original video. The whole original-frame span of a slow-motion
    group is filled so occasional missed hand detections do not shift the prompt.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if vocab_size < VOCAB_SIZE:
        raise ValueError("vocab_size is too small for the Cued Speech vocabulary")
    if len(recognition_results) != len(slow_motion_groups):
        raise ValueError(
            "Recognition result count must equal slow-motion group count: "
            f"{len(recognition_results)} != {len(slow_motion_groups)}"
        )

    frame_indices = np.asarray(valid_frame_indices, dtype=np.int64)
    if frame_indices.ndim != 1:
        raise ValueError("valid_frame_indices must be one-dimensional")
    if frame_indices.size and (
        frame_indices.min() < 0 or frame_indices.max() >= total_frames
    ):
        raise ValueError("valid_frame_indices contains an out-of-range frame index")

    prompt = np.zeros((total_frames, vocab_size), dtype=np.float32)
    for result, group in zip(recognition_results, slow_motion_groups):
        compact_indices = np.asarray(group, dtype=np.int64)
        if compact_indices.size == 0:
            raise ValueError("slow-motion groups must not contain empty groups")
        if compact_indices.min() < 0 or compact_indices.max() >= frame_indices.size:
            raise ValueError("slow-motion group references a missing hand frame")

        position, shape = _labels(result)
        original_indices = frame_indices[compact_indices]
        start, stop = int(original_indices.min()), int(original_indices.max()) + 1
        phonemes = POSITION_PHONEMES[position] + SHAPE_PHONEMES[shape]
        prompt[start:stop, [TOKEN_TO_ID[token] for token in phonemes]] = 1.0
    return prompt
