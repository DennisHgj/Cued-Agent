from __future__ import annotations

import unittest
from types import SimpleNamespace

from cued_agent.postprocess import _parse_json_object, self_correct


VALID_CONTENT = """{
    "Processed_Cued_Speech_Sequence": "n i - h ao",
    "Pinyin_Sequence": "ni hao",
    "Mandarin_Sequence": "你好",
    "Reasoning_Process": "No correction needed."
}"""


class _FakeCompletions:
    def __init__(self, content=VALID_CONTENT, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content, reasoning_content="")
        choice = SimpleNamespace(
            message=message,
            finish_reason=self.finish_reason,
        )
        return SimpleNamespace(choices=[choice])


def _fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class PostprocessTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        value = _parse_json_object('prefix\n```json\n{"Mandarin_Sequence":"你好"}\n```')
        self.assertEqual(value["Mandarin_Sequence"], "你好")

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            _parse_json_object("[]")

    def test_deepseek_request_and_response_contract(self):
        completions = _FakeCompletions()
        result = self_correct(
            "n i - h ao",
            client=_fake_client(completions),
            max_tokens=2048,
        )

        self.assertEqual(result["Mandarin_Sequence"], "你好")
        self.assertEqual(completions.kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(
            completions.kwargs["response_format"], {"type": "json_object"}
        )
        self.assertEqual(completions.kwargs["max_tokens"], 2048)
        self.assertNotIn("temperature", completions.kwargs)
        self.assertIn("JSON", completions.kwargs["messages"][0]["content"])

    def test_rejects_truncated_deepseek_response(self):
        completions = _FakeCompletions(finish_reason="length")
        with self.assertRaisesRegex(RuntimeError, "length"):
            self_correct("n i", client=_fake_client(completions))

    def test_rejects_empty_deepseek_response(self):
        completions = _FakeCompletions(content="")
        with self.assertRaisesRegex(RuntimeError, "empty content"):
            self_correct("n i", client=_fake_client(completions))

    def test_rejects_missing_deepseek_fields(self):
        completions = _FakeCompletions(content='{"Mandarin_Sequence":"你好"}')
        with self.assertRaisesRegex(ValueError, "missing fields"):
            self_correct("n i", client=_fake_client(completions))


if __name__ == "__main__":
    unittest.main()
