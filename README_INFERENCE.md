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

The checked-in `ckpt/` directory does not contain a trained model.

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
- missing `hydra`, `torch`, or `mediapipe`: install `requirements.txt` in the
  active Python environment.
