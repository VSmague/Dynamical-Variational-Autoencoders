import torch
from torch.utils.data import Dataset
import os

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

        # Calcul de la moyenne et de l'écart-type sur tous les mels
        all_mels = [torch.load(f) for f in self.files]  # liste de [seq_len, feat_dim]
        all_mels = torch.cat(all_mels, dim=0)           # concat sur seq_len
        self.mean = all_mels.mean(dim=0)
        self.std = all_mels.std(dim=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = torch.load(self.files[idx])
        return mel
