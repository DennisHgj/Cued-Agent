from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from hand_recognition_agent.CustomizedPromptTemplate import (
    Frame,
    HandRecognition,
    generate_recognition_single,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        parsed = HandRecognition(
            recog_results=[
                Frame(
                    frame_id=0,
                    hand_position=2,
                    hand_shape=6,
                    reasoning_process="test",
                )
            ]
        )
        message = SimpleNamespace(parsed=parsed, refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(total_tokens=123),
        )


class HandRecognitionPayloadTests(unittest.TestCase):
    def test_payload_uses_current_image_url_content_items(self):
        completions = _FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        results, usage = generate_recognition_single(
            ["dGVzdA=="],
            support_set_path=str(ROOT / "hand_recognition_agent" / "support_set"),
            client=client,
        )

        self.assertEqual(results[0]["hand_shape"], 6)
        self.assertEqual(usage, 123)
        content = completions.kwargs["messages"][1]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        self.assertGreater(len(image_items), 1)
        self.assertTrue(
            all(
                item["image_url"]["url"].startswith("data:image/jpeg;base64,")
                for item in image_items
            )
        )


if __name__ == "__main__":
    unittest.main()
