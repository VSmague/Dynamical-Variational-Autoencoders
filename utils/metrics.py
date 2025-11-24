import torch
import soundfile as sf
import numpy as np
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


def sisdr_score(ref: np.ndarray, est: np.ndarray):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between reference and estimated signals.
    ref: reference signal
    est: estimated signal
    return: SI-SDR value in dB
    """
    eps = np.finfo(est.dtype).eps
    ref_energy = np.sum(ref ** 2)
    scale = np.sum(ref * est) / (ref_energy + eps)
    e_target = scale * ref
    e_res = scale * ref - est

    return 10 * np.log10((np.sum(e_target ** 2) + eps) / (np.sum(e_res ** 2) + eps))


class EvalMetrics():
    """
    Evaluation metrics for audio signals.
    """
    def __init__(self, metric='all'):
        self.metric = metric
        self.sisdr_metric = ScaleInvariantSignalDistortionRatio()

    def __call__(self, ref_path, est_path):
        x_ref, fs_ref = sf.read(ref_path)
        x_est, fs_est = sf.read(est_path)

        if len(x_est.shape) > 1:
            x_est = x_est[:, 0]
        if len(x_ref.shape) > 1:
            x_ref = x_ref[:, 0]

        # align
        min_len = np.min(len(x_ref), len(x_est))
        x_ref = x_ref[:min_len]
        x_est = x_est[:min_len]

        if self.metric == 'all':
            pesq_val = pesq(x_ref, x_est, fs_est)
            stoi_val = stoi(x_ref, x_est, fs_est, extended=True)
            sisdr_val = sisdr_score(x_ref, x_est)
            sisdr_val2 = self.sisdr_metric(x_ref, x_est)
            return {'pesq': pesq_val, 'estoi': stoi_val, 'sisdr': sisdr_val, 'sisdr2': sisdr_val2}
        elif self.metric == 'pesq':
            pesq_val = pesq(x_ref, x_est, fs_est)
            return {'pesq': pesq_val}
        elif self.metric == 'estoi':
            stoi_val = stoi(x_ref, x_est, fs_est, extended=True)
            return {'estoi': stoi_val}
        elif self.metric == 'stoi':
            stoi_val = stoi(x_ref, x_est, fs_est, extended=False)
            return {'stoi': stoi_val}
        elif self.metric == 'sisdr':
            sisdr_val = sisdr_score(x_ref, x_est)
            return {'sisdr': sisdr_val}
        else:
            raise ValueError(f"Unknown metric: {self.metric}. Available: all, pesq, estoi, stoi, sisdr.")