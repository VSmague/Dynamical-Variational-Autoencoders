import torch

def test_reconstruction(model, dataset, device="cuda"):
    model.eval()
    x = dataset[0].unsqueeze(0).to(device)   # (1, 200, 80)
    
    with torch.no_grad():
        out = model(x)  # dépend de ton implémentation (ex: returns recon, kl)

    # on suppose que ton model retourne un dict
    # out["x_recon"] : (1, 200, 80)
    # out["kl"]      : (1, 200, dim_z)

    recon = out["x_recon"].cpu().squeeze(0)  # (200,80)

    return x.cpu().squeeze(0), recon, out

import matplotlib.pyplot as plt

def show_mel_vs_recon(mel, recon):
    fig, ax = plt.subplots(1, 2, figsize=(12,4))

    ax[0].imshow(mel.T, aspect="auto", origin="lower")
    ax[0].set_title("Original mel")

    ax[1].imshow(recon.T, aspect="auto", origin="lower")
    ax[1].set_title("Reconstruction VRNN")

    plt.show()
