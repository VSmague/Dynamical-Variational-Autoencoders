import torch


def evaluate(model, dataloader, device='cpu'):
    model.eval()
    total_recon = 0
    total_kld = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.size(1)
            recon, kld = model(batch)
            recon = recon / batch_size
            kld = kld / batch_size
            total_recon += recon.item()
            total_kld += kld.item()
    return total_recon, total_kld


def train(model, dataloader, val_loader, epochs=30, lr=1e-3, kl_anneal_epochs=10, device='cpu'):
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_recon = 0
        total_kld = 0
        
        beta = min(1.0, epoch / kl_anneal_epochs)

        for batch in dataloader:
            batch = batch.to(device)#batch.permute(1, 0, 2).to(device)   # (seq, batch, feat)
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

            total_recon_val, total_kld_val = evaluate(model, val_loader)
    
            train_losses.append(total_recon + beta * total_kld)
            val_losses.append(total_recon_val + beta * total_kld_val)

        print(f"[{epoch+1}] Train: {train_losses[-1]:.3f} | Val: {val_losses[-1]:.3f} | β={beta:.2f}")

        torch.save(model.state_dict(), f"vrnn_epoch{epoch+1}.pt")
