import torch


def evaluate(model, dataloader, device='cpu'):
    model.eval()
    total_recon = 0
    total_kld = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)  # shape: (batch, seq, feat)
            batch_size, seq_len, _ = batch.shape
            
            recon, kld = model(batch)

            recon = recon / (batch_size * seq_len)
            kld = kld / (batch_size * seq_len)

            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1

    return total_recon / n_batches, total_kld / n_batches


def train(model, dataloader, val_loader, epochs=30, lr=1e-3, kl_anneal_epochs=10, device='cpu'):
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_recon = 0
        total_kld = 0
        n_batches = 0

        beta = min(1.0, epoch / kl_anneal_epochs)

        for batch in dataloader:
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape

            optim.zero_grad()
            recon, kld = model(batch)

            recon = recon / (batch_size * seq_len)
            kld = kld / (batch_size * seq_len)

            loss = recon + beta * kld
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1

        train_recon = total_recon / n_batches
        train_kld = total_kld / n_batches

        val_recon, val_kld = evaluate(model, val_loader, device)

        print(f"[{epoch+1}] "
              f"Train: {train_recon + beta*train_kld:.3f} | "
              f"Val: {val_recon + beta*val_kld:.3f} | "
              f"β={beta:.2f}")

        torch.save(model.state_dict(), f"vrnn_epoch{epoch+1}.pt")

# def evaluate(model, dataloader, device='cpu'):
#     model.eval()
#     total_recon = 0
#     total_kld = 0
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(device)
#             batch_size = batch.size(1)
#             recon, kld = model(batch)
#             recon = recon / batch_size
#             kld = kld / batch_size
#             total_recon += recon.item()
#             total_kld += kld.item()
#     return total_recon, total_kld


# def train(model, dataloader, val_loader, epochs=30, lr=1e-3, kl_anneal_epochs=10, device='cpu'):
#     optim = torch.optim.Adam(model.parameters(), lr=lr)

#     train_losses = []
#     val_losses = []

#     for epoch in range(epochs):
#         total_recon = 0
#         total_kld = 0
        
#         beta = min(1.0, epoch / kl_anneal_epochs)

#         for batch in dataloader:
#             model.train()
#             batch = batch.to(device)#batch.permute(1, 0, 2).to(device)   # (seq, batch, feat)
#             batch_size = batch.size(1)
#             seq_len = batch.size(0)

#             optim.zero_grad()
#             recon, kld = model(batch)
#             recon = recon / (batch_size * seq_len)
#             kld = kld / (batch_size * seq_len)
#             loss = recon + beta * kld
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optim.step()

#             total_recon += recon.item()
#             total_kld += kld.item()

#             total_recon_val, total_kld_val = evaluate(model, val_loader, device=device)
#             model.train()
            
#             train_losses.append(total_recon + beta * total_kld)
#             val_losses.append(total_recon_val + beta * total_kld_val)

#         print(f"[{epoch+1}] Train: {train_losses[-1]:.3f} | Val: {val_losses[-1]:.3f} | β={beta:.2f}")

#         torch.save(model.state_dict(), f"vrnn_epoch{epoch+1}.pt")
