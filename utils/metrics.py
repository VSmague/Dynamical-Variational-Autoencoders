import torch
import numpy as np
from pesq import pesq
from pystoi import stoi

def si_sdr(est, ref, eps=1e-8):
    """
    Scale-Invariant SDR (Le Roux et al. 2019)
    est, ref: 1D torch tensors
    """
    ref_energy = torch.sum(ref ** 2) + eps
    scale = torch.sum(ref * est) / ref_energy
    proj = scale * ref
    e_noise = est - proj
    ratio = torch.sum(proj ** 2) / (torch.sum(e_noise ** 2) + eps)
    return 10 * torch.log10(ratio + eps)

def pesq_score(ref, est, sr=16000):
    # PESQ expects numpy arrays; 'wb' for wideband
    return pesq(sr, ref.cpu().numpy(), est.cpu().numpy(), 'wb')

def estoi_score(ref, est, sr=16000):
    return stoi(ref.cpu().numpy(), est.cpu().numpy(), sr, extended=True)