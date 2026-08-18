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
The Reasoning_Process field must contain a brief correction summary, not hidden
chain-of-thought. Example JSON shape:
{"Processed_Cued_Speech_Sequence":"n i - h ao","Pinyin_Sequence":"ni hao",
"Mandarin_Sequence":"你好","Reasoning_Process":"No phoneme change needed."}
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
            candidate = block[4:] if block.startswith("json") else block
            candidate = candidate.strip()
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
    max_tokens: int | None = None,
    client: Any | None = None,
) -> dict[str, str]:
    """Correct a phoneme sequence and convert it to pinyin and Mandarin."""
    if not cued_sequence.strip():
        raise ValueError("Cued Speech sequence must not be empty")

    if client is None:
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
            base_url=base_url
            or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    token_limit = max_tokens
    if token_limit is None:
        raw_limit = os.getenv("DEEPSEEK_MAX_TOKENS", "4096")
        try:
            token_limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError("DEEPSEEK_MAX_TOKENS must be a positive integer") from exc
    if token_limit <= 0:
        raise ValueError("DEEPSEEK_MAX_TOKENS must be a positive integer")

    response = client.chat.completions.create(
        model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(sequence=cued_sequence)},
        ],
        response_format={"type": "json_object"},
        max_tokens=token_limit,
    )
    if not getattr(response, "choices", None):
        raise RuntimeError("P2W API returned no completion choices")
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason not in {None, "stop"}:
        raise RuntimeError(f"P2W API did not finish normally: {finish_reason}")
    message = choice.message
    content = message.content or ""
    if not content.strip():
        raise RuntimeError("P2W API returned empty content")
    result = _parse_json_object(content)

    required_fields = (
        "Processed_Cued_Speech_Sequence",
        "Pinyin_Sequence",
        "Mandarin_Sequence",
        "Reasoning_Process",
    )
    missing_fields = [name for name in required_fields if name not in result]
    if missing_fields:
        raise ValueError("P2W response missing fields: " + ", ".join(missing_fields))
    invalid_fields = [name for name in required_fields if not isinstance(result[name], str)]
    if invalid_fields:
        raise ValueError("P2W response fields must be strings: " + ", ".join(invalid_fields))

    normalized = {
        "Processed_Cued_Speech_Sequence": result["Processed_Cued_Speech_Sequence"],
        "Pinyin_Sequence": result["Pinyin_Sequence"],
        "Mandarin_Sequence": result["Mandarin_Sequence"],
        "Reasoning_Process": result["Reasoning_Process"]
        or getattr(message, "reasoning_content", "")
        or "",
    }
    return normalized
