from __future__ import annotations

import unittest

from cued_agent.postprocess import _parse_json_object


class PostprocessTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        value = _parse_json_object('prefix\n```json\n{"Mandarin_Sequence":"你好"}\n```')
        self.assertEqual(value["Mandarin_Sequence"], "你好")

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            _parse_json_object("[]")


if __name__ == "__main__":
    unittest.main()
