import torch
from torch.utils.data import DataLoader
from model.vrnn import VRNN
from torch.nn.utils.rnn import pad_sequence

from dataset import AudioDataset


def collate_fn(batch):
    # batch: list de tensors [seq_len_i, feat]
    batch = [b for b in batch]  # list de [seq_len, feat]
    padded = pad_sequence(batch, batch_first=False)  # (seq_len_max, batch, feat)
    return padded

def train(model, dataloader, epochs=30, lr=1e-3, kl_anneal_epochs=10):
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_recon = 0
        total_kld = 0
        
        beta = min(1.0, epoch / kl_anneal_epochs)

        for batch in dataloader:
            batch = batch.to(device) #batch.permute(1, 0, 2).to(device)   # (seq, batch, feat)
            batch_size = batch.size(1)

            optim.zero_grad()
            recon, kld = model(batch)
            recon = recon / batch_size
            kld = kld / batch_size
            loss = recon + beta * kld
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_recon += recon.item()
            total_kld += kld.item()

        print(f"[{epoch+1}] Recon: {total_recon:.3f} | KL: {total_kld:.3f} | β={beta:.2f}")

        torch.save(model.state_dict(), f"vrnn_epoch{epoch+1}.pt")
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VRNN().to(device)

    dataset = AudioDataset("data_flac/")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    train(model, dataloader)
