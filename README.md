# Cued-Agent

Cued-Agent is the multi-agent Mandarin Automatic Cued Speech Recognition system
described in the ACM Multimedia 2025 paper. It converts a Cued Speech video into
a phoneme sequence and, optionally, a natural Mandarin sentence.

![Cued-Agent framework](framework.png)

## What is trained

The code now follows the paper's training boundary explicitly:

| Component | Training in this repository | Input used for training |
| --- | --- | --- |
| Lip visual encoder | Yes | Lip ROI frames |
| CTC head + attention decoder | Yes, jointly with the lip encoder | Phoneme labels |
| MLLM Hand Recognition Agent | No | Prompting + visual support set |
| Hand Prompt Decoding Agent | No additional training or parameters | Adds `4.5 * H` to CTC logits during inference |
| Self-Correction P2W Agent | No | LLM prompting |

In other words, “decoder training” means the attention decoder and CTC head are
trained with the lip model. The later hand-prompt fusion step reuses that trained
decoder and is parameter-free.

## Repository layout

```text
cued_agent/                              maintained inference package
hand_recognition_agent/                  OpenAI vision prompt + support set
lip_agent_and_prompt_decoding_agent/     lip/decoder model, training, ESPnet code
self-p2w-agent/                          legacy experiment scripts
util/                                    ROI preprocessing helpers
tests/                                   repository-owned unit and CLI tests
run_inference.py                         single-video CLI
batch_inference.py                       batch CLI
Inference.py                             backward-compatible import
```

## Installation

Python 3.8-3.11 and an NVIDIA GPU are supported for inference. The original
server environment (`auto_avsr`, Python 3.8) is part of the release validation;
Python 3.10 or 3.11 is recommended when creating a new environment.

The training entry point translates trainer defaults between PyTorch Lightning
1.5 and 2.x, so the original `auto_avsr` environment and newly created
environments use the same command.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if the default pip
build is not suitable. Copy `.env.example` values into your shell environment;
the code reads keys from the environment and does not store secrets in source.

The repository does not currently distribute the research dataset or a trained
checkpoint. See `ckpt/README.md` for the required checkpoint contents.

## End-to-end inference

Full four-stage inference requires `OPENAI_API_KEY` and `DEEPSEEK_API_KEY`:

```bash
python run_inference.py \
  --video HS-0001.mp4 \
  --checkpoint ckpt/lip_decoder.ckpt \
  --output outputs/HS-0001.json
```

The paper defaults are used automatically: hand prompt weight `4.5`, joint beam
search CTC weight `0.5`, and beam size `20`.

For reproducible/offline hand decoding, pass previously saved hand labels:

```bash
python run_inference.py \
  --video HS-0001.mp4 \
  --checkpoint ckpt/lip_decoder.ckpt \
  --hand-results path/to/hand_results.json \
  --no-self-correction
```

For a pure-lip ablation with no API calls:

```bash
python run_inference.py \
  --video HS-0001.mp4 \
  --checkpoint ckpt/lip_decoder.ckpt \
  --lip-only \
  --no-self-correction
```

See `README_INFERENCE.md` for stage contracts and output details.

## Train the lip encoder and decoder

The label loader accepts either:

- four CSV fields for lip training: `dataset,video,input_length,token_ids`;
- six fields for hand-prompt evaluation: the four fields above plus
  `hand_result_json,hand_position_npy`.

Run from the model directory and override the local dataset paths:

```bash
cd lip_agent_and_prompt_decoding_agent
python train_lip_agent.py \
  data.dataset.root_dir=/path/to/dataset \
  data.dataset.label_dir=labels \
  data.dataset.train_file=train.csv \
  data.dataset.val_file=val.csv \
  pretrained_model_path=/path/to/base_visual_model.pth \
  exp_dir=../exp \
  exp_name=lip_decoder
```

This trains the visual encoder, CTC projection, and attention decoder with joint
CTC/attention loss. It does not load or train hand-recognition data.

## Validation

Fast repository-owned tests do not require API keys or model weights:

```bash
python -m unittest discover -s tests -v
```

Release 1.1.0 was also validated on the original research server with Python
3.8.20, PyTorch 2.0.1+cu117, PyTorch Lightning 1.5.10, and an RTX A6000:

- all 7 repository tests passed;
- pure-lip inference completed on `HS-0001.mp4` with the 2.87 GB
  `H_multi_lip.ckpt`;
- offline hand-prompt fusion completed with 8 detected slow-motion groups;
- one real CCS training batch and one validation batch completed after loading
  762/767 compatible tensors from the base visual checkpoint.

The public repository does not distribute the research checkpoint or dataset,
so those full checks require separately supplied assets. API-backed stages also
require network access and valid credentials; they are never replaced with
fabricated labels.

## Citation

```bibtex
@inproceedings{huang2025cuedagent,
  title={Cued-Agent: A Multi-Agent Framework for Automatic Cued Speech Recognition},
  author={Huang, Guanjie and others},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2025}
}
```

## License

MIT. See `LICENSE`.
