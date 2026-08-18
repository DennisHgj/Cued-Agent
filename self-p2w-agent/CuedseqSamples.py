"""Optional few-shot example builder for P2W experiments."""

from __future__ import annotations

from pathlib import Path


def build_cuedseq_list(train_label_path, mandarin_path, sentence_num):
    label_lines = Path(train_label_path).read_text(encoding="utf-8").splitlines()
    mandarin_lines = Path(mandarin_path).read_text(encoding="utf-8").splitlines()
    count = min(int(sentence_num), len(label_lines))
    content_list = [
        "Here are Mandarin Cued Speech sequences and corresponding sentences."
    ]
    for line in label_lines[:count]:
        fields = line.split(",")
        if len(fields) < 3:
            raise ValueError("Training label rows must be comma-separated")
        cued_sequence = fields[-1].strip()
        video_num = int(Path(fields[1]).stem.split("-")[-1])
        if not 1 <= video_num <= len(mandarin_lines):
            raise IndexError(f"Sentence index out of range: {video_num}")
        mandarin_raw = mandarin_lines[video_num - 1].strip()
        mandarin = mandarin_raw.split("：", 1)[-1].strip()
        content_list.append(
            f"Cued Speech sequence: {cued_sequence}; Mandarin sentence: {mandarin}"
        )
    return content_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build optional P2W few-shot examples")
    parser.add_argument("train_label_path")
    parser.add_argument("mandarin_path")
    parser.add_argument("--sentence-num", type=int, default=10)
    args = parser.parse_args()
    print(
        build_cuedseq_list(
            args.train_label_path, args.mandarin_path, args.sentence_num
        )
    )
