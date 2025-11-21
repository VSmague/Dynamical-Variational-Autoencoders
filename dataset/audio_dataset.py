import torch
from torch.utils.data import Dataset
import os


class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # charge directement le Mel-spectrogram
        mel = torch.load(self.files[idx])  # shape: [seq_len, feat_dim]
        return mel
