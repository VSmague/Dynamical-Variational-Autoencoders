import torch
from torch.utils.data import Dataset
import os

class AudioDataset(Dataset):
    def __init__(self, data_dirs):
        self.files = []
        for d in data_dirs:
            self.files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.pt')]

        # Calcul de la moyenne et de l'écart-type sur tous les mels
        all_mels = [torch.load(f) for f in self.files]
        all_mels = torch.cat(all_mels, dim=0)
        self.mean = all_mels.mean(dim=0)
        self.std = all_mels.std(dim=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = torch.load(self.files[idx])
        return mel