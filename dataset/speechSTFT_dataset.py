import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Dataset PyTorch : STFT Power
# -----------------------------
class SpeechSTFTDataset(Dataset):
    """
    Dataset pour transformer des fichiers WAV en séquences STFT power spectrogram
    Chaque séquence est de taille (seq_len, 513) correspondant aux positive frequencies
    """
    def __init__(self, wav_paths, seq_len=50, sample_rate=16000):
        self.wav_paths = wav_paths
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.n_fft = 1024
        self.hop_length = 256
        self.window = torch.hann_window(self.n_fft)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        # Charger WAV et resampler
        wav, sr = torchaudio.load(self.wav_paths[idx])
        wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav[0]  # si stéréo, prendre le canal 0

        # Calculer STFT
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )  # shape: (freq_bins, time_frames)

        # Spectrogramme de puissance
        power = stft.abs()**2
        power = power[:self.n_fft//2 + 1, :]  # garder les fréquences positives (513)

        # Découpage / padding pour séquence de longueur fixe
        T = power.shape[1]
        if T < self.seq_len:
            pad = self.seq_len - T
            power = torch.cat([power, torch.zeros(power.shape[0], pad)], dim=1)
        else:
            power = power[:, :self.seq_len]

        # Transposer pour avoir (seq_len, freq_bins)
        return power.transpose(0, 1)  # (seq_len, 513)


# -----------------------------
# Fonction utilitaire pour récupérer les fichiers WAV
# -----------------------------
def get_wav_paths(data_dir):
    wav_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_paths.append(os.path.join(root, f))
    return sorted(wav_paths)


# -----------------------------
# Exemple d'utilisation
# -----------------------------
if __name__ == "__main__":
    data_dir = "path_to_your_wav_dataset"  # changer avec le chemin de ton dataset
    wav_paths = get_wav_paths(data_dir)

    # Créer dataset et dataloader
    seq_len = 50
    batch_size = 16

    dataset = SpeechSTFTDataset(wav_paths, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Tester une batch
    for batch in dataloader:
        # batch.shape -> (batch_size, seq_len, 513)
        print(batch.shape)
        break
