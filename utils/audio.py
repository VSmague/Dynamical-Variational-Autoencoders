import torchaudio
import torch

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

def wav_to_logmel(wav):
    mel = mel_transform(wav)
    mel = torch.log(mel + 1e-6)
    return mel.transpose(0, 1)  # (frames, 80)
