"""Compatibility wrapper for the maintained P2W implementation.

New code should call ``cued_agent.postprocess.self_correct`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from cued_agent.postprocess import self_correct


def single_process(predicted_cued_seq):
    """Return the normalized result dictionary for one phoneme sequence."""
    return self_correct(predicted_cued_seq)


def file_loop(predicted_txt, _label_csv, output_folder):
    """Process a text file containing one predicted sequence per line."""
    source = Path(predicted_txt)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    sequences = source.read_text(encoding="utf-8").splitlines()
    for index, sequence in enumerate(sequences):
        if not sequence.strip():
            continue
        result = single_process(sequence.strip())
        (output / f"{index:06d}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-correct one Cued Speech sequence")
    parser.add_argument("sequence")
    args = parser.parse_args()
    print(json.dumps(single_process(args.sequence), ensure_ascii=False, indent=2))
