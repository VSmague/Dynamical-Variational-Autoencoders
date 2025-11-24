import os
import json
from typing import Optional

import torch
from torch import Tensor
from model.VRNN import VRNN


def save_checkpoint(state, checkpoint_dir, name="last.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, name)
    torch.save(state, path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    history = checkpoint["history"]
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return start_epoch, history


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = VRNN()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def isd_loss(x: Tensor, y: Tensor):
    y = y + torch.finfo(y.dtype).eps
    div = x / y
    return torch.sum(div - torch.log(div) - 1)


def kld_loss(z_mean: Tensor, z_logvar: Tensor, z_mean_p: Optional[Tensor]=None, z_logvar_p: Optional[Tensor]=None):
    if z_mean_p is None:
        z_mean_p = torch.zeros_like(z_mean)
    if z_logvar_p is None:
        z_logvar_p = torch.zeros_like(z_logvar)

    return -0.5 * torch.sum(z_logvar - z_logvar_p
                            - torch.div(torch.exp(z_logvar) + (z_mean - z_mean_p) ** 2,
                                        torch.exp(z_logvar_p) + torch.finfo(z_logvar.dtype).eps))


def evaluate(model: VRNN, dataloader, device):
    model.eval()
    total_recon, total_kl = 0, 0
    n_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)              # (batch, seq, feat)
            batch = batch.permute(1, 0, 2)        # → (seq, batch, feat)
            seq_len, batch_size, _ = batch.shape

            recon = torch.exp(model(batch))
            recon_loss = isd_loss(batch, recon)
            kl_loss = kld_loss(model.z_mean, model.z_logvar, model.z_mean_p, model.z_logvar_p)
            recon_loss = recon_loss / (batch_size * seq_len)
            kl_loss = kl_loss / (batch_size * seq_len)

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    return total_recon / n_batches, total_kl / n_batches


def train(
    model: VRNN,
    train_loader,
    val_loader,
    checkpoint_dir="checkpoints",
    resume=False,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    kl_anneal_epochs=10,
    patience=7,
    save_frequency=5,
    device="cuda"
):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tracking
    history = {
        "train": [],
        "train_recon": [],
        "train_kl": [],
        "val": [],
        "val_recon": [],
        "val_kl": [],
        "beta": [],
    }

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume training
    last_ckpt = os.path.join(checkpoint_dir, "last.pt")
    if resume and os.path.exists(last_ckpt):
        start_epoch, history = load_checkpoint(model, optimizer, last_ckpt, device)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    n_batches = len(train_loader)
    for epoch in range(start_epoch, epochs):
        model.train()
        total_recon, total_kl = 0, 0
        beta = min(1.0, epoch / kl_anneal_epochs)

        for batch in train_loader:
            batch = batch.to(device)             # (batch, seq, feat)
            batch = batch.permute(1, 0, 2)       # → (seq, batch, feat)
            seq_len, batch_size, _ = batch.shape

            optimizer.zero_grad()
            recon = torch.exp(model(batch))
            recon_loss = isd_loss(batch, recon)
            mean, logvar = model(batch)
            kl_loss = kld_loss(model.z_mean, model.z_logvar, model.z_mean_p, model.z_logvar_p)

            recon_loss = recon_loss / (batch_size * seq_len)
            kl_loss = kl_loss / (batch_size * seq_len)
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        train_recon = total_recon / n_batches
        train_kl = total_kl / n_batches
        train_loss = train_recon + beta * train_kl

        # Validation
        val_recon, val_kl = evaluate(model, val_loader, device=device)
        val_loss = val_recon + beta * val_kl

        print(f"[{epoch+1}/{epochs}] "
              f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | β={beta:.3f}")

        # Update history
        history["train"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        history["beta"].append(beta)

        # Save last checkpoint
        if epoch % save_frequency == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history
            }, checkpoint_dir, "last.pt")

        # Save best model
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history
            }, checkpoint_dir, "best.pt")
            patience_counter = 0
            print(" → Saved BEST model")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save history as JSON
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete.")
    return history
