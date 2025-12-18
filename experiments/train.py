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




#### modifications ajout scheduler learning rate

# MODIFICATION  : Ajout de l'argument 'scheduler' pour recharger son état
def load_checkpoint_s(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    # On charge l'état du scheduler s'il existe dans la sauvegarde
    if "scheduler_state" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        
    start_epoch = checkpoint["epoch"] + 1
    history = checkpoint["history"]
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return start_epoch, history

def train_model_with_scheduler(
    model,
    train_loader,
    val_loader,
    checkpoint_dir="checkpoints",
    resume=False,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    kl_anneal_epochs=10,
    patience=20, 
    device="cuda"
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # MODIFICATION 2 : Initialisation du Scheduler
    # factor=0.5 : Divise le learning rate par 2 quand ça stagne
    # patience=5 : Attend 5 epochs de stagnation avant d'agir
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Tracking
    history = {
        "train": [], "val": [], "beta": [],
        "train_recon": [], "train_kld": [],
        "val_recon": [], "val_kld": [],
        "lr": [] # MODIFICATION : On garde une trace du Learning Rate
    }

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume training
    last_ckpt = os.path.join(checkpoint_dir, "last.pt")
    if resume and os.path.exists(last_ckpt):
        # MODIFICATION : On passe le scheduler à la fonction de chargement
        start_epoch, history = load_checkpoint_s(model, optimizer, scheduler, last_ckpt, device)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_recon, total_kld = 0, 0
        n_batches = 0

        # Calcul de Beta (KL Annealing)
        beta = min(1.0, epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else 1.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            batch = batch.to(device)             
            batch = batch.permute(1, 0, 2)       
            seq_len, batch_size, _ = batch.shape

            optimizer.zero_grad()
            _,recon, kld = model(batch)

            recon = recon / (batch_size * seq_len)
            kld = kld / (batch_size * seq_len)
            loss = recon + beta * kld

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1

        train_recon = total_recon / n_batches
        train_kld = total_kld / n_batches
        train_loss = train_recon + beta * train_kld

        # Validation
        val_recon, val_kld = evaluate(model, val_loader, device=device)
        val_loss = val_recon + beta * val_kld

        # MODIFICATION 3 : Step du Scheduler
        # On informe le scheduler de la nouvelle validation loss
        scheduler.step(val_loss)
        
        # Récupération du LR actuel pour l'affichage
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch+1}/{epochs}] "
              f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | "
              f"β={beta:.3f} | LR={current_lr:.2e}")

        # Update history
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["beta"].append(beta)
        history["train_recon"].append(train_recon)
        history["train_kld"].append(train_kld)
        history["val_recon"].append(val_recon)
        history["val_kld"].append(val_kld)
        history["lr"].append(current_lr) # On loggue le LR

        # Save Checkpoint Dict (Helper)
        # MODIFICATION 4 : On ajoute l'état du scheduler dans la sauvegarde
        checkpoint_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), # <--- ICI
            "history": history
        }

        # Save last checkpoint
        save_checkpoint(checkpoint_dict, checkpoint_dir, "last.pt")

        # Save best model
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_dict, checkpoint_dir, "best.pt")
            patience_counter = 0
            print(" → Saved BEST model")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # Save history as JSON
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        # Helper pour rendre les float32 sérialisables en JSON si besoin
        class FloatEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.Tensor): return obj.item()
                return json.JSONEncoder.default(self, obj)
        json.dump(history, f, indent=4, cls=FloatEncoder)

    print("\nTraining complete.")
    return history














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




def evaluate(model, dataloader, device):
    model.eval()
    total_recon, total_kld = 0, 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)              # (batch, seq, feat)
            batch = batch.permute(1, 0, 2)        # → (seq, batch, feat)
            seq_len, batch_size, _ = batch.shape

            _,recon, kld = model(batch)
            recon = recon / (batch_size * seq_len)
            kld = kld / (batch_size * seq_len)

            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1

    return total_recon / n_batches, total_kld / n_batches


def train_model(
    model,
    train_loader,
    val_loader,
    checkpoint_dir="checkpoints",
    resume=False,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    kl_anneal_epochs=10,
    patience=7,
    device="cuda"
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Tracking
    history = {
        "train": [],
        "val": [],
        "beta": [],
        "train_recon": [],
        "train_kld": [],
        "val_recon": [],
        "val_kld": []
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
    for epoch in range(start_epoch, epochs):
        model.train()
        total_recon, total_kld = 0, 0
        n_batches = 0

        beta = min(1.0, epoch / kl_anneal_epochs)

        # Wrap train_loader with tqdm for a progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"): # Added tqdm here
            batch = batch.to(device)             # (batch, seq, feat)
            batch = batch.permute(1, 0, 2)       # → (seq, batch, feat)
            seq_len, batch_size, _ = batch.shape

            optimizer.zero_grad()
            _, recon, kld = model(batch)

            recon = recon / (batch_size * seq_len)
            kld = kld / (batch_size * seq_len)
            loss = recon + beta * kld

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1

        train_recon = total_recon / n_batches
        train_kld = total_kld / n_batches
        train_loss = train_recon + beta * train_kld

        # Validation
        val_recon, val_kld = evaluate(model, val_loader, device=device)
        val_loss = val_recon + beta * val_kld

        print(f"[{epoch+1}/{epochs}] "
              f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | β={beta:.3f}")

        # Update history
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["beta"].append(beta)
        history["train_recon"].append(train_recon)
        history["train_kld"].append(train_kld)
        history["val_recon"].append(val_recon)
        history["val_kld"].append(val_kld)

        # Save last checkpoint
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
























# def save_checkpoint(state, checkpoint_dir, name="last.pt"):
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     path = os.path.join(checkpoint_dir, name)
#     torch.save(state, path)


# def load_checkpoint(model, optimizer, checkpoint_path, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state"])
#     optimizer.load_state_dict(checkpoint["optimizer_state"])
#     start_epoch = checkpoint["epoch"] + 1
#     history = checkpoint["history"]
#     print(f"Checkpoint loaded from: {checkpoint_path}")
#     return start_epoch, history


# def evaluate(model, dataloader, device):
#     model.eval()
#     total_recon, total_kld = 0, 0
#     n_batches = 0
   
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(device)              # (batch, seq, feat)
#             batch = batch.permute(1, 0, 2)        # → (seq, batch, feat)
#             seq_len, batch_size, _ = batch.shape

#             _,recon, kld = model(batch)
#             recon = recon / (batch_size * seq_len)
#             kld = kld / (batch_size * seq_len)

#             total_recon += recon.item()
#             total_kld += kld.item()
#             n_batches += 1

#     return total_recon / n_batches, total_kld / n_batches


# def train(
#     model,
#     train_loader,
#     val_loader,
#     checkpoint_dir="checkpoints",
#     resume=False,
#     epochs=50,
#     batch_size=32,
#     lr=1e-4,
#     kl_anneal_epochs=10,
#     patience=7,
#     device="cuda"
# ):

#     device = torch.device(device if torch.cuda.is_available() else "cpu")
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # Tracking
#     history = {
#         "train": [],
#         "val": [],
#         "beta": []
#     }

#     start_epoch = 0
#     best_val_loss = float("inf")
#     patience_counter = 0

#     # Resume training
#     last_ckpt = os.path.join(checkpoint_dir, "last.pt")
#     if resume and os.path.exists(last_ckpt):
#         start_epoch, history = load_checkpoint(model, optimizer, last_ckpt, device)
#         print(f"Resuming from epoch {start_epoch}")

#     # Training loop
#     for epoch in range(start_epoch, epochs):
#         model.train()
#         total_recon, total_kld = 0, 0
#         n_batches = 0

#         beta = min(1.0, epoch / kl_anneal_epochs)

#         for batch in train_loader:
#             batch = batch.to(device)             # (batch, seq, feat)
#             batch = batch.permute(1, 0, 2)       # → (seq, batch, feat)
#             seq_len, batch_size, _ = batch.shape

#             optimizer.zero_grad()
#             _,recon, kld = model(batch)

#             recon = recon / (batch_size * seq_len)
#             kld = kld / (batch_size * seq_len)
#             loss = recon + beta * kld

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#             total_recon += recon.item()
#             total_kld += kld.item()
#             n_batches += 1

#         train_recon = total_recon / n_batches
#         train_kld = total_kld / n_batches
#         train_loss = train_recon + beta * train_kld

#         # Validation
#         val_recon, val_kld = evaluate(model, val_loader, device=device)
#         val_loss = val_recon + beta * val_kld

#         print(f"[{epoch+1}/{epochs}] "
#               f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | β={beta:.3f}")

#         # Update history
#         history["train"].append(train_loss)
#         history["val"].append(val_loss)
#         history["beta"].append(beta)

#         # Save last checkpoint
#         save_checkpoint({
#             "epoch": epoch,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "history": history
#         }, checkpoint_dir, "last.pt")

#         # Save best model
#         if val_loss < best_val_loss - 1e-4:
#             best_val_loss = val_loss
#             save_checkpoint({
#                 "epoch": epoch,
#                 "model_state": model.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "history": history
#             }, checkpoint_dir, "best.pt")
#             patience_counter = 0
#             print(" → Saved BEST model")
#         else:
#             patience_counter += 1

#         # Early stopping
#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break

#     # Save history as JSON
#     with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
#         json.dump(history, f, indent=4)

#     print("\nTraining complete.")
#     return history
