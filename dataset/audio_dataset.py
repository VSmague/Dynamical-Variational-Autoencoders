import torch
from torch.utils.data import Dataset
import os


class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        all_mels = [torch.load(f) for f in self.files]
        all_mels = torch.cat(all_mels, dim=0)
        self.mean = all_mels.mean(dim=0)
        self.std = all_mels.std(dim=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # charge directement le Mel-spectrogram
        mel = torch.load(self.files[idx])  # shape: [seq_len, feat_dim]
        return mel
