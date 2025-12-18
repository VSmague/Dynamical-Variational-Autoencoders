import os
import sys
import json
import glob

sys.path.append(os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import IPython.display as ipd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


import torch
import torchaudio
from torch.utils.data import random_split, DataLoader


from model.VRNN import VRNN, VRNN_Student
from dataset.audio_dataset import AudioDataset
from utils.padding import collate_fn
from experiments.train import *


# Fonction locale pour préparer l'image (Dénormalisation + Transpose)
def prep_for_plot(tensor, mean, std):
    denorm = tensor * (std + 1e-6) + mean
    return denorm.numpy().T # (Freq, Time) pour Librosa

        
# Fonction Helper : Mel -> Audio
def mel_to_audio_playable(mel_tensor, mean, std, name):
    # Sécurité CPU
    mel_tensor = mel_tensor.detach().cpu()
    mean = mean.cpu()
    std = std.cpu()
    
    # A. Dénormalisation
    mel_denorm = mel_tensor * (std + 1e-6) + mean
    
    # B. Conversion Numpy & Transpose (Seq, Feat) -> (Feat, Seq)
    mel_numpy = mel_denorm.numpy()
    if mel_numpy.shape[0] > mel_numpy.shape[1]:
        mel_numpy = mel_numpy.T
    
    # C. dB to Power & Griffin-Lim
    mel_power = librosa.db_to_power(mel_numpy)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=22050, n_fft=1024, hop_length=256, win_length=1024
    )
    
    # E. Sauvegarde et Affichage
    filename = f"{name}.wav"
    sf.write(filename, audio, 22050)
    display(ipd.Audio(filename))