import torch
from torch.utils.data import Dataset
import torchaudio
import os

class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_rate=16000):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.flac')]
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)  # waveform: [channels, time]
        
        # Si audio stéréo, prendre seulement 1 canal
        if waveform.size(0) > 1:
            waveform = waveform[0:1, :]
        
        # Resample si nécessaire
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Transformer en spectrogramme
        if self.transform:
            spec = self.transform(waveform)  # ex: MelSpectrogram
        else:
            # spectrogramme simple par défaut
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=80,
                n_fft=1024,
                hop_length=256
            )(waveform)
        
        # shape: [1, n_mels, time] -> on veut [time, n_mels]
        spec = spec.squeeze(0).transpose(0, 1)

        # normalisation par spectrogramme
        spec = (spec - spec.mean()) / (spec.std() + 1e-9)
        
        return spec
