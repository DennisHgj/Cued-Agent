import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "lip_agent_and_prompt_decoding_agent"
sys.path.insert(0, str(MODEL_ROOT))

from training_contract import (  # noqa: E402
    validate_compatible_fraction,
    validate_training_contract,
)


def make_config(**overrides):
    values = {
        "pretrained_model_path": None,
        "resume_from_checkpoint": None,
        "allow_random_initialization": False,
        "min_pretrained_tensor_fraction": 0.99,
        "data": SimpleNamespace(
            include_hand=False,
            max_frames=2000,
            max_frames_val=1600,
            num_workers=4,
        ),
        "trainer": SimpleNamespace(accumulate_grad_batches=1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainingContractTests(unittest.TestCase):
    def test_reproduction_run_rejects_implicit_random_initialization(self):
        with self.assertRaisesRegex(ValueError, "requires pretrained_model_path"):
            validate_training_contract(make_config())

    def test_random_control_requires_explicit_opt_in(self):
        cfg = make_config(allow_random_initialization=True)
        self.assertEqual(validate_training_contract(cfg), "random")

    def test_pretrained_and_resume_paths_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.ckpt"
            checkpoint.touch()
            pretrained = make_config(pretrained_model_path=str(checkpoint))
            resumed = make_config(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(validate_training_contract(pretrained), "pretrained")
            self.assertEqual(validate_training_contract(resumed), "resume")

        missing = make_config(pretrained_model_path="missing-model.pth")
        with self.assertRaises(FileNotFoundError):
            validate_training_contract(missing)

    def test_lip_training_rejects_hand_inputs_and_invalid_frame_budget(self):
        hand_data = SimpleNamespace(
            include_hand=True,
            max_frames=2000,
            max_frames_val=1600,
            num_workers=4,
        )
        with self.assertRaisesRegex(ValueError, "include_hand"):
            validate_training_contract(
                make_config(data=hand_data, allow_random_initialization=True)
            )

        invalid_data = SimpleNamespace(
            include_hand=False,
            max_frames=0,
            max_frames_val=1600,
            num_workers=4,
        )
        with self.assertRaisesRegex(ValueError, "max_frames"):
            validate_training_contract(
                make_config(data=invalid_data, allow_random_initialization=True)
            )

    def test_checkpoint_compatibility_threshold(self):
        self.assertAlmostEqual(
            validate_compatible_fraction(762, 767, 0.99), 762 / 767
        )
        with self.assertRaisesRegex(ValueError, "below the required threshold"):
            validate_compatible_fraction(700, 767, 0.99)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_compatible_fraction(762, 767, 1.1)


if __name__ == "__main__":
    unittest.main()
