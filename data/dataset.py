import torch
from torch.utils.data import Dataset
import numpy as np
from channels import resize_channel, build_three_channels
from synthetic import generate_clip


class ThermalDataset(Dataset):
    def __init__(self, n_samples=1000, pos_ratio=0.2, clip_len=16, img_size=112):
        self.clip_len = clip_len
        self.img_size = img_size
        n_pos = int(n_samples * pos_ratio)
        n_neg = n_samples - n_pos
        self.samples = (
                [(i, 1) for i in range(n_pos)] +
                [(i + n_pos, 0) for i in range(n_neg)]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seed, label = self.samples[idx]
        clip = generate_clip(seed, label)
        T_ch, T_smooth, delta_T = build_three_channels(clip)
        T_ch = resize_channel(T_ch, self.img_size)
        T_smooth = resize_channel(T_smooth, self.img_size)
        delta_T = resize_channel(delta_T, self.img_size)
        tensor = np.stack([T_ch, T_smooth, delta_T], axis=0)
        return (
            torch.tensor(tensor, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )
