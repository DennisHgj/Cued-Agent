from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from hand_recognition_agent.CustomizedPromptTemplate import (
    Frame,
    HandRecognition,
    _structured_completion_parser,
    generate_recognition_single,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeCompletions:
    def __init__(self, frame_id=0):
        self.kwargs = None
        self.frame_id = frame_id

    def parse(self, **kwargs):
        self.kwargs = kwargs
        parsed = HandRecognition(
            recog_results=[
                Frame(
                    frame_id=self.frame_id,
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
        self.assertEqual(completions.kwargs["model"], "gpt-4.1")
        self.assertIs(completions.kwargs["response_format"], HandRecognition)
        content = completions.kwargs["messages"][1]["content"]
        image_items = [item for item in content if item["type"] == "image_url"]
        self.assertGreater(len(image_items), 1)
        self.assertTrue(
            all(
                item["image_url"]["url"].startswith("data:image/jpeg;base64,")
                for item in image_items
            )
        )
        self.assertTrue(
            all(item["image_url"]["detail"] == "high" for item in image_items)
        )

    def test_rejects_invalid_image_detail(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions())
        )
        with self.assertRaisesRegex(ValueError, "OPENAI_IMAGE_DETAIL"):
            generate_recognition_single(
                ["dGVzdA=="],
                support_set_path=str(ROOT / "hand_recognition_agent" / "support_set"),
                image_detail="ultra",
                client=client,
            )

    def test_rejects_out_of_order_frame_ids(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(frame_id=1))
        )
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            generate_recognition_single(
                ["dGVzdA=="],
                support_set_path=str(ROOT / "hand_recognition_agent" / "support_set"),
                client=client,
            )

    def test_schema_rejects_out_of_range_hand_labels(self):
        with self.assertRaises(ValueError):
            Frame(
                frame_id=0,
                hand_position=5,
                hand_shape=8,
                reasoning_process="invalid",
            )

    def test_uses_openai_1x_beta_parser_fallback(self):
        completions = _FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace()),
            beta=SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            ),
        )
        parser = _structured_completion_parser(client)
        self.assertEqual(parser.__self__, completions)

    def test_rejects_sdk_without_structured_parser(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace())
        )
        with self.assertRaisesRegex(RuntimeError, "structured parsing"):
            _structured_completion_parser(client)


if __name__ == "__main__":
    unittest.main()
