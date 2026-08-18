from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import httpx
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on optional live API dependency
    httpx = None
    OpenAI = None

from cued_agent.postprocess import self_correct
from hand_recognition_agent.CustomizedPromptTemplate import generate_recognition_single


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(OpenAI is None or httpx is None, "openai/httpx not installed")
class ApiHttpContractTests(unittest.TestCase):
    def test_openai_sdk_serializes_images_and_parses_structured_response(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            response_content = json.dumps(
                {
                    "recog_results": [
                        {
                            "frame_id": 0,
                            "hand_position": 2,
                            "hand_shape": 6,
                            "reasoning_process": "mock transport",
                        }
                    ]
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-hand-contract",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4.1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_content,
                                "refusal": None,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as http_client:
            client = OpenAI(api_key="test-key", http_client=http_client)
            results, usage = generate_recognition_single(
                ["dGVzdA=="],
                support_set_path=str(ROOT / "hand_recognition_agent" / "support_set"),
                client=client,
            )

        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured["body"]["model"], "gpt-4.1")
        self.assertEqual(captured["body"]["response_format"]["type"], "json_schema")
        image_items = [
            item
            for item in captured["body"]["messages"][1]["content"]
            if item["type"] == "image_url"
        ]
        self.assertTrue(image_items)
        self.assertTrue(
            all(item["image_url"]["detail"] == "high" for item in image_items)
        )
        self.assertEqual(results[0]["hand_shape"], 6)
        self.assertEqual(usage, 2)

    def test_deepseek_sdk_serializes_json_mode_and_parses_response(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            response_content = json.dumps(
                {
                    "Processed_Cued_Speech_Sequence": "n i - h ao",
                    "Pinyin_Sequence": "ni hao",
                    "Mandarin_Sequence": "你好",
                    "Reasoning_Process": "No correction needed.",
                },
                ensure_ascii=False,
            )
            return httpx.Response(
                200,
                json={
                    "id": "deepseek-p2w-contract",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "deepseek-v4-pro",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_content,
                                "reasoning_content": "mock reasoning",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as http_client:
            client = OpenAI(
                api_key="test-key",
                base_url="https://api.deepseek.com",
                http_client=http_client,
            )
            result = self_correct(
                "n i - h ao",
                client=client,
                max_tokens=2048,
            )

        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["body"]["model"], "deepseek-v4-pro")
        self.assertEqual(
            captured["body"]["response_format"], {"type": "json_object"}
        )
        self.assertEqual(captured["body"]["max_tokens"], 2048)
        self.assertNotIn("temperature", captured["body"])
        self.assertEqual(result["Mandarin_Sequence"], "你好")


if __name__ == "__main__":
    unittest.main()
