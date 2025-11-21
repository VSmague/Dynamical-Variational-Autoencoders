import os
import torch
import torchaudio
from torch.utils.data import Dataset
from utils.audio import wav_to_logmel


class VCTKDataset(Dataset):
    def __init__(self, root, seq_len=200):
        self.wav_paths = []
        self.seq_len = seq_len

        speakers = sorted(os.listdir(root))
        for spk in speakers:
            spk_path = os.path.join(root, spk)
            for fname in os.listdir(spk_path):
                if fname.endswith(".wav"):
                    self.wav_paths.append(os.path.join(spk_path, fname))

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.wav_paths[idx])
        wav = torchaudio.functional.resample(wav, sr, 16000)

        mel = wav_to_logmel(wav[0])  # (frames, 80)
        T = mel.shape[0]

        if T < self.seq_len:
            pad = self.seq_len - T
            mel = torch.cat([mel, torch.zeros(pad, 80)], dim=0)
        else:
            mel = mel[:self.seq_len]

        return mel  # (seq_len, 80)
