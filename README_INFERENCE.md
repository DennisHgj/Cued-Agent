# Inference guide

## Stage contracts

1. Video preprocessing returns lip ROI tensor `[T, 1, 88, 88]`, compact hand ROI
   frames, hand positions, and zero-based indices back into the original video.
2. Hand Recognition selects slow-motion groups with the paper thresholds
   `sigma=6` and `theta=2`, then predicts one position and shape per group.
3. The prompt builder maps compact detected-hand frames back to original frames
   and creates `H` with shape `[T, 44]`.
4. The trained lip encoder produces `L`. The trained CTC projection produces
   `L'`; inference uses `L' + lambda_prompt * H` only in the CTC scorer. The
   attention scorer continues to use lip features.
5. Optional P2W self-correction converts the decoded phonemes to pinyin and a
   Mandarin sentence.

No API failure is silently converted into a fabricated hand label. Use
`--lip-only`, `--hand-results`, or `--no-self-correction` when a stage should be
disabled explicitly.

## Required assets

- a fine-tuned Lightning checkpoint containing lip encoder, CTC head, and
  attention decoder weights;
- `hand_recognition_agent/support_set/` for live hand recognition;
- API keys in environment variables for live API stages.

The default live API contract is:

- `OPENAI_HAND_MODEL=gpt-4.1` and `OPENAI_IMAGE_DETAIL=high` for fine-grained
  hand-shape recognition;
- `DEEPSEEK_MODEL=deepseek-v4-pro` and `DEEPSEEK_MAX_TOKENS=4096` for P2W;
- DeepSeek JSON Output is requested explicitly and malformed, empty, or
  truncated responses fail instead of being silently accepted.

See `.env.example` for all configurable values. The paper-era
`gpt-4o-2024-08-06` snapshot remains an explicit opt-in only while the provider
keeps it available.

The hand-recognition client prefers the current
`client.chat.completions.parse` entry point and falls back to
`client.beta.chat.completions.parse` for the OpenAI 1.x SDK used by the original
`auto_avsr` environment.

The checked-in `ckpt/` directory does not contain a trained model.

The maintainer release check uses the original `auto_avsr` environment and a
private research checkpoint. The isolated training server cannot call external
LLM APIs, so live hand recognition and P2W must be verified on a networked host;
offline hand-result fusion is covered on the server.

## Commands

```bash
python run_inference.py --help
python batch_inference.py --help
```

Full pipeline:

```bash
python run_inference.py --video VIDEO.mp4 --checkpoint MODEL.ckpt
```

Phoneme-only, pure-lip smoke run:

```bash
python run_inference.py \
  --video VIDEO.mp4 \
  --checkpoint MODEL.ckpt \
  --lip-only \
  --no-self-correction \
  --device cpu
```

CPU execution is supported for debugging but is not practical for normal
Conformer inference.

## Precomputed hand results

`--hand-results` accepts either a JSON list or an object containing
`recog_results`. There must be one item for each automatically selected
slow-motion group:

```json
{
  "recog_results": [
    {"frame_id": 0, "hand_position": 2, "hand_shape": 6},
    {"frame_id": 1, "hand_position": 3, "hand_shape": 0}
  ]
}
```

`hand_gesture` remains accepted as a legacy alias for `hand_shape`.

## Output

The JSON result includes raw beam-search phonemes, the optional corrected
sequence, pinyin, Mandarin, and inference metadata. When self-correction is
disabled, pinyin and Mandarin are empty strings.

## Common failures

- `checkpoint not found`: the repository does not ship the trained weight file;
  pass an explicit existing `--checkpoint`.
- `Checkpoint keys do not match`: use a checkpoint trained with the 44-token
  Mandarin Cued Speech vocabulary and the checked-in model configuration.
- `OPENAI_API_KEY is required`: use `--lip-only` or `--hand-results`, or export
  the key.
- `DEEPSEEK_API_KEY is required`: use `--no-self-correction`, or export the key.
- `P2W API did not finish normally`: retry the request or increase
  `DEEPSEEK_MAX_TOKENS` when the finish reason is `length`.
- missing `hydra`, `torch`, or `mediapipe`: install `requirements.txt` in the
  active Python environment.
