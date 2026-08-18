"""Self-correction phoneme-to-word stage using an OpenAI-compatible API."""

from __future__ import annotations

import json
import os
from typing import Any


SYSTEM_PROMPT = """You post-process Mandarin Cued Speech phoneme sequences.
Return one JSON object with exactly these string fields:
Processed_Cued_Speech_Sequence, Pinyin_Sequence, Mandarin_Sequence,
Reasoning_Process. Preserve the input as much as possible and make only the
minimum changes needed for valid Cued Speech combinations and fluent Mandarin.
"""

USER_PROMPT = """Cued Speech words are separated by '-' and phonemes inside a
word are separated by spaces. A word normally contains zero to two consonant
phonemes and exactly one vowel phoneme. Common visual confusions include
b/p/m, t/n/r, j/q/x, g/k/h, z/c/s, sh/s, ch/c, and zh/z. Vowels are less likely
to be wrong. Convert the sequence to pinyin and a natural Mandarin sentence,
while changing as few phonemes as possible.

Input sequence:
{sequence}
"""


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if "```" in text:
        blocks = text.split("```")
        for block in blocks:
            candidate = block.removeprefix("json").strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
    if not text.startswith("{"):
        start, stop = text.find("{"), text.rfind("}")
        if start >= 0 and stop > start:
            text = text[start : stop + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("P2W response must be a JSON object")
    return value


def self_correct(
    cued_sequence: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Correct a phoneme sequence and convert it to pinyin and Mandarin."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required for self-correction. "
            "Use --no-self-correction to run through phoneme decoding only."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("The openai Python package is required for P2W") from exc

    client = OpenAI(
        api_key=key,
        base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    response = client.chat.completions.create(
        model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(sequence=cued_sequence)},
        ],
        temperature=0.2,
    )
    message = response.choices[0].message
    content = message.content or ""
    result = _parse_json_object(content)

    normalized = {
        "Processed_Cued_Speech_Sequence": str(
            result.get("Processed_Cued_Speech_Sequence", cued_sequence)
        ),
        "Pinyin_Sequence": str(result.get("Pinyin_Sequence", "")),
        "Mandarin_Sequence": str(result.get("Mandarin_Sequence", "")),
        "Reasoning_Process": str(
            result.get(
                "Reasoning_Process", getattr(message, "reasoning_content", "") or ""
            )
        ),
    }
    return normalized
