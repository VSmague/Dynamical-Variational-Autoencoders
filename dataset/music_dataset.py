import os
import json

import numpy as np
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


def get_nsynth_paths(dataset_root, instrument_family=None):
    """
    Parses NSynth metadata and returns paths for a specific instrument family.
    
    Args:
        dataset_root: Path to the unzipped folder (containing 'audio' and 'examples.json')
        instrument_family: String ID of the family to keep (e.g., 'keyboard', 'guitar').
                           If None, returns all files.
                           
    Families:
        bass, brass, flute, guitar, keyboard, mallet, 
        organ, reed, string, synth_lead, vocal
    """
    json_path = os.path.join(dataset_root, "examples.json")
    audio_dir = os.path.join(dataset_root, "audio")

    with open(json_path, "r") as f:
        metadata = json.load(f)

    valid_paths = []
    count = 0

    for filename, info in metadata.items():
        # Check if file exists
        wav_path = os.path.join(audio_dir, f"{filename}.wav")
        if not os.path.exists(wav_path):
            continue

        # Filter by family
        if instrument_family:
            if info["instrument_family_str"] != instrument_family:
                continue

        valid_paths.append(wav_path)
        count += 1

    return valid_paths


# -----------------------------
# Dataset PyTorch : STFT Power
# -----------------------------
class MusicDataset(Dataset):
    """
    Dataset to transform WAV files into STFT power spectrogram sequences
    Each sequence has size (seq_len, 513) corresponding to positive frequencies
    -----------------------------
    Args:
        wav_paths (list): list of paths to WAV files
        seq_len (int): length of sequences (in STFT frames)
        sample_rate (int): target sample rate for audio files
        win_len_sec (float): window length in seconds for STFT
        trim (bool): whether to trim silence from the audio
        shuffle (bool): whether to shuffle the dataset sequences
    -----------------------------
    Returns:
        STFT power spectrogram sequence of shape (seq_len, 513)
    """
    def __init__(self, wav_paths, seq_len=50, sample_rate=16_000, win_len_sec=64e-3, trim=False, shuffle=True):
        self.wav_paths = wav_paths
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.trim = trim

        # STFT parameters
        # self.win_length = int(np.power(2, np.ceil(np.log2(win_len_sec * sample_rate))))  # next power of 2
        # self.hop_length = int(0.25 * self.win_length)
        self.win_length = 1024
        self.hop_length = 256
        self.n_fft = self.win_length
        self.window = torch.sin(torch.arange(0.5, self.win_length + 0.5) / self.win_length * torch.pi)

        self.samples_per_seq = (self.seq_len - 1) * self.hop_length

        self.valid_wavs = []
        # self.compute_len(shuffle)


    def compute_len(self, shuffle):
        for wavfile in self.wav_paths:
            wav, sr = torchaudio.load(wavfile)

            if wav.shape[0] > 1:
                wav = wav[0]  # if stereo, get the first channel
            wav = wav.squeeze(0)  # if stereo, get the first channel
            if self.sample_rate != sr:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

            if self.trim:
               _, (start, end) = librosa.effects.trim(wav.numpy(), top_db=30)
            else:
                start, end = 0, len(wav)

            file_len = end - start
            n_seq = (1 + int(file_len / self.hop_length)) // self.seq_len
            for i in range(n_seq):
                seq_start = start + i * self.samples_per_seq
                self.valid_wavs.append((wavfile, seq_start, sr))

            if shuffle:
                np.random.shuffle(self.valid_wavs)


    def __len__(self):
        # return len(self.valid_wavs)
        return len(self.wav_paths)


    def __getitem__(self, idx):
        # wavfile, start, sr = self.valid_wavs[idx]
        # wav, _ = torchaudio.load(wavfile, frame_offset=start, num_frames=self.samples_per_seq)
        wav, sr = torchaudio.load(self.wav_paths[idx])
        wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav[0]  # if stereo, get the first channel

        # wav = wav.squeeze(0)
        wav = wav / torch.max(torch.abs(wav))  # normalization

        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )

        power = stft.abs() ** 2
        power = power[:self.n_fft//2 + 1, :]

        T = power.shape[1]
        if T < self.seq_len:
            pad = self.seq_len - T
            power = torch.cat([power, torch.zeros(power.shape[0], pad)], dim=1)
        else:
            power = power[:, :self.seq_len]

        return power.transpose(0, 1)  # (seq_len, 513)

        power = stft.abs().pow(2).transpose(0, 1)
        return power


if __name__ == "__main__":
    dataset_root = "data/nsynth/nsynth-test"  # Change to your NSynth dataset path
    families= ["bass", "brass", "flute", "guitar", "keyboard", "mallet", 
        "organ", "reed", "string", "synth_lead", "vocal"]
    for fam in families:
        paths = get_nsynth_paths(dataset_root, fam)
        print(f"{fam}: {len(paths)} samples")
    instrument_family = None  # change to desired instrument family or None for all
    paths = get_nsynth_paths(dataset_root, instrument_family)
    print(f"All samples: {len(paths)} samples")

    instrument_family = "guitar"
    wav_paths = get_nsynth_paths(dataset_root, instrument_family=instrument_family)
    seq_len = 50
    batch_size = 16
    dataset = MusicDataset(wav_paths, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Test a batch
    print(f"For {instrument_family} instrument:")
    print(len(dataloader), "batches in the dataloader.")
    for batch in dataloader:
        # batch.shape -> (batch_size, seq_len, 513)
        print(batch.shape)
        break