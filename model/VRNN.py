import torch
import torch.nn as nn
from torch import Tensor


def get_activation(name):
    if name is None:
        return None
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "linear":
        return None   # activation linéaire = pas d'activation
    raise ValueError(f"Activation inconnue: {name}")


class MLP(nn.Module):
    def __init__(self, input_dim, n_list, f_list, dropout=0.0):
        """
        input_dim : dimension d'entrée
        n_list    : [n1, n2, ..., nL]
        f_list    : ["relu", "sigmoid", "tanh", "linear"]
        """
        super().__init__()

        assert len(n_list) == len(f_list), "n_list et f_list doivent avoir la même longueur"

        layers = []
        prev_dim = input_dim

        for n, f_name in zip(n_list, f_list):
            # couche linéaire
            layers.append(nn.Linear(prev_dim, n))

            # activation
            act = get_activation(f_name)
            if act is not None:
                layers.append(act)

            # dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = n

        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)


class VRNN(nn.Module):
    def __init__(self, x_dim=80, h_dim=256, z_dim=32, phi_x_dim=32, phi_z_dim=16, dropout=0.0):
        super().__init__()
        self.x_dim = x_dim # 513
        self.h_dim = h_dim # 128
        self.z_dim = z_dim # 16
        self.num_rnn = 1

        # Features extractors phi
        self.phi_x = MLP(
            input_dim=x_dim,
            n_list=[256, phi_x_dim],
            f_list=["tanh", "linear"],
            dropout=dropout,
        )

        self.phi_z = MLP(
            input_dim=z_dim,
            n_list=[32, 64, phi_z_dim],
            f_list=["tanh", "tanh", "linear"],
            dropout=dropout,
        )

        # Encoder q(z|x,h)
        self.enc = MLP(
            input_dim=phi_x_dim + h_dim,
            n_list=[256],
            f_list=["tanh"],
            dropout=dropout
        )
        self.enc_mu = nn.Linear(256, z_dim)
        self.enc_logvar = nn.Linear(256, z_dim)

        # Prior p(z|h)
        self.prior = MLP(
            input_dim=h_dim,
            n_list=[256],
            f_list=["tanh"],
            dropout=dropout
        )
        self.prior_mu = nn.Linear(256, z_dim)
        self.prior_logvar = nn.Linear(256, z_dim)

        # Decoder p(x|z,h)
        self.dec = MLP(
            input_dim=phi_z_dim + h_dim,
            n_list=[256],
            f_list=["tanh"],
            dropout=dropout
        )
        self.dec_mu = nn.Linear(256, x_dim)
        self.dec_logvar = nn.Linear(256, x_dim)

        # RNN
        self.rnn = nn.LSTM(phi_x_dim + phi_z_dim, h_dim, num_layers=1)

        self.z_mean = torch.zeros((1, 1, self.z_dim))
        self.z_logvar = torch.zeros((1, 1, self.z_dim))
        self.z_mean_p = None
        self.z_logvar_p = None


    def sample_norm(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return torch.addcmul(mean, eps, std)


    def gen_x(self, phi_z_t, h_t):
        dec_input = torch.cat((phi_z_t, h_t), 2)
        dec_output = self.dec(dec_input)
        logvar = self.dec_logvar(dec_output)
        return logvar


    def gen_z(self, h):
        prior_output = self.prior(h)
        mean_prior = self.prior_mu(prior_output)
        logvar_prior = self.prior_logvar(prior_output)
        return mean_prior, logvar_prior


    def encode(self, phi_x_t, h_t):
        enc_input = torch.cat((phi_x_t, h_t), 2)
        enc_output = self.enc(enc_input)
        mean_zt = self.enc_mu(enc_output)
        logvar_zt = self.enc_logvar(enc_output)
        return mean_zt, logvar_zt


    def recurrence(self, phi_x_t, phi_z_t, h_t, c_t):
        rnn_input = torch.cat((phi_x_t, phi_z_t), -1)
        _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))
        return h_tp1, c_tp1


    def forward(self, x: Tensor):
        # (seq_len, batch_size, x_dim)
        seq_len, batch_size, _ = x.shape
        device = x.device

        self.z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(device)
        self.z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(device)
        y_logvar = torch.zeros((seq_len, batch_size, self.x_dim)).to(device)
        self.z = torch.zeros((seq_len, batch_size, self.z_dim)).to(device)
        h = torch.zeros((seq_len, batch_size, self.h_dim)).to(device)
        z_t = torch.zeros(batch_size, self.z_dim).to(device)
        h_t = torch.zeros(self.num_rnn, batch_size, self.h_dim).to(device)
        c_t = torch.zeros(self.num_rnn, batch_size, self.h_dim).to(device)

        phi_x = self.phi_x(x)
        for t in range(seq_len):
            phi_xt = phi_x[t, :, :].unsqueeze(0)
            h_t_last = h_t.view(self.num_rnn, 1, batch_size, self.h_dim)[-1, :, :, :]

            mean_zt, logvar_zt = self.encode(phi_xt, h_t_last)
            z_t = self.sample_norm(mean_zt, logvar_zt)
            phi_zt = self.phi_z(z_t)

            y_t_mean, y_t_logvar = self.gen_x(phi_zt, h_t_last)

            self.z_mean[t, :, :] = mean_zt
            self.z_logvar[t, :, :] = logvar_zt
            self.z[t, :, :] = torch.squeeze(z_t)

            y_logvar[t, :, :] = torch.squeeze(y_t_logvar)
            h[t, :, :] = torch.squeeze(h_t_last)
            h_t, c_t = self.recurrence(phi_xt, phi_zt, h_t, c_t) # recurrence for t+1

        self.z_mean_p, self.z_logvar_p = self.gen_z(h)
        return y_logvar


    def forward2(self, x: Tensor):
        """
        x : (seq_len, batch, x_dim)
        """
        seq_len, batch, _ = x.size()
        h = torch.zeros(1, batch, self.h_dim, device=x.device)

        kld_loss = 0
        recon_loss = 0

        for t in range(seq_len):
            x_t = x[t]
            phi_x_t = self.phi_x(x_t)

            # Prior
            prior_h = self.prior(h.squeeze(0))
            prior_mu = self.prior_mu(prior_h)
            prior_logvar = self.prior_logvar(prior_h)

            # Encoder q(z|x,h)
            enc_h = self.enc(torch.cat([phi_x_t, h.squeeze(0)], dim=1))
            enc_mu = self.enc_mu(enc_h)
            enc_logvar = self.enc_logvar(enc_h)

            # Sampling z
            std = torch.exp(0.5 * enc_logvar)
            eps = torch.randn_like(std)
            z_t = enc_mu + eps * std
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_h = self.dec(torch.cat([phi_z_t, h.squeeze(0)], dim=1))
            dec_mu = self.dec_mu(dec_h)
            dec_logvar = self.dec_logvar(dec_h)

            # Loss
            recon_loss += 0.5 * torch.sum(
                dec_logvar
                + (x_t - dec_mu)**2 / torch.exp(dec_logvar)
            )

            # KL divergence KL(q(z|x,h) || p(z|h))
            kld_loss += 0.5 * torch.sum(
                prior_logvar - enc_logvar
                + (torch.exp(enc_logvar) + (enc_mu - prior_mu)**2) / torch.exp(prior_logvar)
                - 1
            )

            # --------- RNN update : input = φ_x(x_t) + φ_z(z_t)
            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)

        return recon_loss, kld_loss
