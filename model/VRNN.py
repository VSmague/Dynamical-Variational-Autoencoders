import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class VRNN(nn.Module):
    def __init__(self, x_dim=80, h_dim=256, z_dim=32):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Encoder q(z|x,h)
        self.enc_mlp = MLP(x_dim + h_dim, 256)
        self.enc_mu = nn.Linear(256, z_dim)
        self.enc_logvar = nn.Linear(256, z_dim)

        # Prior p(z|h)
        self.prior_mlp = MLP(h_dim, 256)
        self.prior_mu = nn.Linear(256, z_dim)
        self.prior_logvar = nn.Linear(256, z_dim)

        # Decoder p(x|z,h)
        self.dec_mlp = MLP(z_dim + h_dim, 256)
        self.dec_out = nn.Linear(256, x_dim)

        # RNN
        self.rnn = nn.GRU(x_dim + z_dim, h_dim)

    def forward(self, x):
        """
        x : (seq_len, batch, x_dim)
        """
        seq_len, batch, _ = x.size()
        h = torch.zeros(1, batch, self.h_dim, device=x.device)

        kld_loss = 0
        recon_loss = 0

        for t in range(seq_len):
            x_t = x[t]

            # Prior
            prior_h = self.prior_mlp(h.squeeze(0))
            prior_mu = self.prior_mu(prior_h)
            prior_logvar = self.prior_logvar(prior_h)

            # Encoder q(z|x,h)
            enc_h = self.enc_mlp(torch.cat([x_t, h.squeeze(0)], dim=1))
            enc_mu = self.enc_mu(enc_h)
            enc_logvar = self.enc_logvar(enc_h)

            # Sampling z
            std = torch.exp(0.5 * enc_logvar)
            eps = torch.randn_like(std)
            z_t = enc_mu + eps * std

            # Decoder
            dec_h = self.dec_mlp(torch.cat([z_t, h.squeeze(0)], dim=1))
            x_pred = self.dec_out(dec_h)

            # Loss
            recon_loss += F.mse_loss(x_pred, x_t, reduction="sum")

            kld = 0.5 * torch.sum(
                prior_logvar - enc_logvar +
                (torch.exp(enc_logvar) + (enc_mu - prior_mu)**2) / torch.exp(prior_logvar)
                - 1
            )
            kld_loss += kld

            # RNN update
            rnn_input = torch.cat([x_t, z_t], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)

        return recon_loss, kld_loss
