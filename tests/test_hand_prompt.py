from __future__ import annotations

import unittest

import numpy as np

from cued_agent.hand_prompt import (
    TOKEN_TO_ID,
    build_hand_prompt,
    find_slow_motion_groups,
    select_keyframes,
)


class HandPromptTests(unittest.TestCase):
    def test_slow_motion_groups_and_keyframes(self):
        positions = np.array(
            [[0, 0], [2, 0], [4, 0], [20, 0], [40, 0], [42, 0]],
            dtype=np.float32,
        )
        groups = find_slow_motion_groups(positions, movement_threshold=6)
        self.assertEqual(groups, [[1, 2], [5]])
        self.assertEqual(select_keyframes(groups), [1, 5])

    def test_prompt_is_aligned_to_original_video_frames(self):
        results = [
            {"hand_position": 0, "hand_shape": 0},
            {"hand_position": 4, "hand_gesture": 7},
        ]
        groups = [[1, 2], [4]]
        # Compact hand frames correspond to sparse original frames.
        frame_indices = np.array([0, 2, 3, 8, 9, 10])
        prompt = build_hand_prompt(results, groups, frame_indices, total_frames=12)

        self.assertEqual(prompt.shape, (12, 44))
        self.assertTrue(np.all(prompt[2:4, TOKEN_TO_ID["an"]] == 1))
        self.assertTrue(np.all(prompt[2:4, TOKEN_TO_ID["p"]] == 1))
        self.assertEqual(prompt[9, TOKEN_TO_ID["eng"]], 1)
        self.assertEqual(prompt[9, TOKEN_TO_ID["sh"]], 1)
        self.assertEqual(float(prompt[:, 43].sum()), 0.0)  # EOS is never prompted.

    def test_result_count_must_match_groups(self):
        with self.assertRaises(ValueError):
            build_hand_prompt([], [[1]], [0, 1], total_frames=2)


if __name__ == "__main__":
    unittest.main()
