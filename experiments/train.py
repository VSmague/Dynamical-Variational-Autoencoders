import torch
from torch.utils.data import DataLoader
from model.vrnn import VRNN

def train(model, dataloader, epochs=30, lr=1e-3):
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_recon = 0
        total_kld = 0
        
        for batch in dataloader:
            batch = batch.permute(1, 0, 2).to(device)   # (seq, batch, feat)

            optim.zero_grad()
            recon, kld = model(batch)
            loss = recon + kld
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_recon += recon.item()
            total_kld += kld.item()

        print(f"[{epoch+1}] Recon: {total_recon:.1f} | KLD: {total_kld:.1f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VRNN().to(device)

    # Ex: dataset = YourSpectrogramDataset(...)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # train(model, dataloader)
