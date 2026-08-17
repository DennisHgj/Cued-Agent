import os

import torch
import torchaudio
import torchvision

from .hand_util import load_hand_recog


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset_CCS(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir,
            label_path,
            subset,
            modality,
            audio_transform,
            video_transform,
            rate_ratio=640,
            include_hand=False,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        self.include_hand = include_hand

        self.list = self.load_list(label_path)

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            fields = path_count_label.split(",")
            if len(fields) == 4:
                dataset_name, rel_path, input_length, token_id = fields
                hand_recog_path = hand_position_path = None
            elif len(fields) == 6:
                dataset_name, rel_path, input_length, token_id, hand_recog_path, hand_position_path = fields
            else:
                raise ValueError(
                    "Each label row must contain 4 fields (lip training) or "
                    f"6 fields (hand-prompt evaluation), got {len(fields)}"
                )
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    torch.tensor([int(_) for _ in token_id.split()]),
                    hand_recog_path,
                    hand_position_path
                )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id, hand_recog_path, hand_position_path = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if os.path.exists(path) is False:
            raise FileNotFoundError(f"{path} does not exist.")
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            sample = {"input": video, "target": token_id}
            if self.include_hand:
                if not hand_recog_path or not hand_position_path:
                    raise ValueError(
                        "Hand-prompt evaluation requires 6-field label rows"
                    )
                if not os.path.exists(hand_recog_path):
                    raise FileNotFoundError(f"{hand_recog_path} does not exist.")
                if not os.path.exists(hand_position_path):
                    raise FileNotFoundError(f"{hand_position_path} does not exist.")
                sample["hand_matrix"] = load_hand_recog(
                    hand_recog_path, hand_position_path, input_length
                )
            return sample
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}
        elif self.modality == "audiovisual":
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            return {"video": video, "audio": audio, "target": token_id}

    def __len__(self):
        return len(self.list)
