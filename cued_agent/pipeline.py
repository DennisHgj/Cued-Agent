"""Paper-aligned end-to-end Cued-Agent inference orchestration."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .hand_prompt import (
    VOCAB_SIZE,
    build_hand_prompt,
    find_slow_motion_groups,
    select_keyframes,
)
from .postprocess import self_correct


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "lip_agent_and_prompt_decoding_agent"


class CuedAgentInference:
    """Run video -> phonemes -> Mandarin with explicit optional API stages."""

    def __init__(
        self,
        cfg: Any,
        *,
        checkpoint_path: str | Path | None = None,
        detector: str = "mediapipe",
        hand_weight: float = 4.5,
        ctc_weight: float = 0.5,
        beam_size: int = 20,
        device: str | None = None,
        use_hand: bool = True,
        use_self_correction: bool = True,
        hand_results_path: str | Path | None = None,
        hand_recognizer: Callable[..., Any] | None = None,
        postprocessor: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        if not 0.0 <= ctc_weight <= 1.0:
            raise ValueError("ctc_weight must be between 0 and 1")
        if hand_weight < 0:
            raise ValueError("hand_weight must be non-negative")
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        if detector != "mediapipe":
            raise ValueError(
                "Only the bundled mediapipe detector is currently implemented"
            )

        self.cfg = cfg
        self.hand_weight = float(hand_weight)
        self.ctc_weight = float(ctc_weight)
        self.beam_size = int(beam_size)
        self.use_hand = use_hand
        self.use_self_correction = use_self_correction
        self.hand_results_path = (
            Path(hand_results_path).expanduser().resolve()
            if hand_results_path
            else None
        )
        self.postprocessor = postprocessor or self_correct
        self.total_frame_num = 0

        self._load_runtime()
        self.hand_recognizer = hand_recognizer or self._default_hand_recognizer
        self._init_video_preprocessor()

        requested_device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested_device}")
        self.device = self.torch.device(requested_device)

        checkpoint = checkpoint_path or getattr(cfg, "ckpt_path", None)
        if not checkpoint:
            raise ValueError(
                "A fine-tuned lip encoder + CTC/attention decoder checkpoint is required"
            )
        self.checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Fine-tuned lip/decoder checkpoint not found: {self.checkpoint_path}"
            )

        self.modelmodule = self.ModelModule(
            cfg,
            hand_weight=self.hand_weight,
            ctc_weight=self.ctc_weight,
            output_results=False,
        )
        self._load_finetuned_checkpoint(self.checkpoint_path)
        self.modelmodule.eval().to(self.device)

        self.support_set_path = PROJECT_ROOT / "hand_recognition_agent" / "support_set"
        if self.use_hand and not self.hand_results_path and not self.support_set_path.is_dir():
            raise FileNotFoundError(f"Hand support set not found: {self.support_set_path}")

    def _load_runtime(self) -> None:
        """Load optional heavy dependencies only when inference is constructed."""
        if str(MODEL_ROOT) not in sys.path:
            sys.path.insert(0, str(MODEL_ROOT))
        try:
            import cv2
            import torch

            from hand_recognition_agent.CustomizedPromptTemplate import (
                generate_recognition_single,
            )
            from lip_agent_and_prompt_decoding_agent.datamodule.transforms import (
                VideoTransform,
            )
            from lip_agent_and_prompt_decoding_agent.lightning_CCS_hand_prompt_decoding import (
                ModelModule_CCS_Hand_prompt_decoding,
                get_beam_search_decoder,
            )
            from lip_hand_seg_CS_latest import single_video_segment
            from util.mediapipe.detector import LandmarksDetector
            from util.mediapipe.video_process import VideoProcess
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise RuntimeError(
                f"Missing inference dependency: {exc.name}. "
                "Install requirements.txt before running inference."
            ) from exc

        self.cv2 = cv2
        self.torch = torch
        self.VideoTransform = VideoTransform
        self.ModelModule = ModelModule_CCS_Hand_prompt_decoding
        self.get_beam_search_decoder = get_beam_search_decoder
        self.single_video_segment = single_video_segment
        self.LandmarksDetector = LandmarksDetector
        self.VideoProcess = VideoProcess
        self._default_hand_recognizer = generate_recognition_single

    def _init_video_preprocessor(self) -> None:
        self.landmarks_detector = self.LandmarksDetector()
        self.video_process = self.VideoProcess(convert_gray=False)
        self.video_transform = self.VideoTransform(subset="test")

    def _load_finetuned_checkpoint(self, checkpoint_path: Path) -> None:
        payload = self.torch.load(str(checkpoint_path), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must contain a state dictionary")

        if isinstance(payload.get("state_dict"), dict):
            state = payload["state_dict"]
            target = self.modelmodule
        elif isinstance(payload.get("model_state_dict"), dict):
            state = payload["model_state_dict"]
            target = self.modelmodule.model
        else:
            state = payload
            target = self.modelmodule.model

        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}

        target_keys = set(target.state_dict())
        matched = target_keys.intersection(state)
        if not matched:
            raise ValueError(
                "Checkpoint keys do not match the configured lip/decoder architecture"
            )
        prefix = "model." if target is self.modelmodule else ""
        required_prefixes = (
            f"{prefix}encoder.",
            f"{prefix}ctc.",
            f"{prefix}decoder.",
        )
        missing_parts = [
            part.rstrip(".")
            for part in required_prefixes
            if not any(key.startswith(part) for key in matched)
        ]
        if missing_parts:
            raise ValueError(
                "Checkpoint is not an end-to-end lip/decoder checkpoint; missing "
                + ", ".join(missing_parts)
            )
        target.load_state_dict(state, strict=False)

    def _load_video_rgb(self, video_path: Path) -> np.ndarray:
        capture = self.cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        frames: list[np.ndarray] = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frames.append(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        return np.stack(frames)

    def preprocess_video(
        self, video_path: str | Path
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        path = Path(video_path).expanduser().resolve()
        video_rgb = self._load_video_rgb(path)
        self.total_frame_num = int(video_rgb.shape[0])

        landmarks = self.landmarks_detector(video_rgb)
        lip_roi = self.video_process(video_rgb, landmarks)
        if lip_roi is None or len(lip_roi) == 0:
            raise RuntimeError("Lip ROI extraction produced no frames")
        lip_video = self.torch.from_numpy(np.asarray(lip_roi)).permute(0, 3, 1, 2)
        lip_video = self.video_transform(lip_video)

        hand_frames, hand_positions, valid_frame_indices = self.single_video_segment(
            str(path)
        )
        return lip_video, hand_frames, hand_positions, valid_frame_indices

    def _load_hand_results(self) -> list[dict[str, Any]]:
        if self.hand_results_path is None:
            raise RuntimeError("No hand results path configured")
        if not self.hand_results_path.is_file():
            raise FileNotFoundError(
                f"Precomputed hand recognition JSON not found: {self.hand_results_path}"
            )
        with self.hand_results_path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        results = value.get("recog_results") if isinstance(value, dict) else value
        if not isinstance(results, list):
            raise ValueError("Hand recognition JSON must be a list or contain recog_results")
        return results

    def hand_recognition(
        self,
        hand_frames: np.ndarray,
        hand_positions: np.ndarray,
        valid_frame_indices: np.ndarray,
    ) -> np.ndarray:
        groups = find_slow_motion_groups(hand_positions)
        if not groups:
            return np.zeros((self.total_frame_num, VOCAB_SIZE), dtype=np.float32)
        keyframe_indices = select_keyframes(groups)

        if self.hand_results_path:
            recognition_results = self._load_hand_results()
        else:
            encoded_frames: list[str] = []
            for index in keyframe_indices:
                if index >= len(hand_frames):
                    raise ValueError("Keyframe index exceeds extracted hand frames")
                ok, buffer = self.cv2.imencode(".jpg", hand_frames[index])
                if not ok:
                    raise RuntimeError(f"Failed to encode hand keyframe {index}")
                encoded_frames.append(base64.b64encode(buffer.tobytes()).decode("ascii"))
            recognition_results, _usage = self.hand_recognizer(
                encoded_frames, support_set_path=str(self.support_set_path)
            )

        return build_hand_prompt(
            recognition_results,
            groups,
            valid_frame_indices,
            self.total_frame_num,
            vocab_size=len(self.modelmodule.token_list),
        )

    def _align_hand_prompt(self, hand_prompt: np.ndarray, target_frames: int):
        prompt = self.torch.as_tensor(
            hand_prompt, dtype=self.torch.float32, device=self.device
        )
        if prompt.ndim != 2 or prompt.shape[1] != len(self.modelmodule.token_list):
            raise ValueError(
                "hand prompt must have shape [T, vocabulary_size], got "
                f"{tuple(prompt.shape)}"
            )
        if prompt.shape[0] != target_frames:
            prompt = self.torch.nn.functional.interpolate(
                prompt.transpose(0, 1).unsqueeze(0),
                size=target_frames,
                mode="nearest",
            ).squeeze(0).transpose(0, 1)
        return prompt

    def lip_hand_joint_decoding(self, lip_video: Any, hand_prompt: np.ndarray) -> str:
        with self.torch.inference_mode():
            inputs = lip_video.unsqueeze(0).to(self.device)
            encoded, _ = self.modelmodule.model.encoder(inputs, None)
            encoded = encoded.squeeze(0)
            aligned_prompt = self._align_hand_prompt(hand_prompt, encoded.shape[0])

            beam_search = self.get_beam_search_decoder(
                self.modelmodule.model,
                self.modelmodule.token_list,
                ctc_weight=self.ctc_weight,
                beam_size=self.beam_size,
            )
            hypotheses = beam_search(encoded, hand_matrix=aligned_prompt)
            if not hypotheses:
                raise RuntimeError("Beam search returned no hypotheses")
            token_ids = [int(value) for value in hypotheses[0].yseq[1:]]
            if token_ids and token_ids[-1] == self.modelmodule.model.eos:
                token_ids.pop()
            predicted = self.modelmodule.text_transform.post_process(
                self.torch.tensor(token_ids, dtype=self.torch.long)
            )
            return predicted.replace("<eos>", "").strip()

    def __call__(self, video_path: str | Path) -> dict[str, Any]:
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        lip_video, hand_frames, hand_positions, frame_indices = self.preprocess_video(path)
        if self.use_hand:
            hand_prompt = self.hand_recognition(
                hand_frames, hand_positions, frame_indices
            )
        else:
            hand_prompt = np.zeros(
                (self.total_frame_num, len(self.modelmodule.token_list)),
                dtype=np.float32,
            )

        raw_sequence = self.lip_hand_joint_decoding(lip_video, hand_prompt)
        if self.use_self_correction:
            result: dict[str, Any] = self.postprocessor(raw_sequence)
        else:
            result = {
                "Processed_Cued_Speech_Sequence": raw_sequence,
                "Pinyin_Sequence": "",
                "Mandarin_Sequence": "",
                "Reasoning_Process": "Self-correction disabled",
            }
        result["Raw_Cued_Speech_Sequence"] = raw_sequence
        result["Inference_Metadata"] = {
            "video": str(path),
            "checkpoint": str(self.checkpoint_path),
            "hand_prompt_enabled": self.use_hand,
            "self_correction_enabled": self.use_self_correction,
            "hand_weight": self.hand_weight,
            "ctc_weight": self.ctc_weight,
            "beam_size": self.beam_size,
            "device": str(self.device),
        }
        return result
