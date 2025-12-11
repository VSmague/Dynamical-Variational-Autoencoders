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
        return None
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
            layers.append(nn.Linear(prev_dim, n))

            act = get_activation(f_name)
            if act is not None:
                layers.append(act)

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = n

        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)


class SRNN(nn.Module):
    def __init__(self, x_dim=513, h_dim=128, g_dim=128, z_dim=16, dropout=0.0):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.num_rnn_h = 1
        self.num_rnn_g = 1

        # x_tm1 -> h_t
        self.phi_x = MLP(
            input_dim=x_dim,
            n_list=[256],
            f_list=["tanh"],
            dropout=dropout
        )
        # h_t, Forward recurrence
        self.rnn_h = nn.LSTM(256, h_dim, num_layers=self.num_rnn_h, batch_first=True)

        # h_t, x_t -> g_t
        self.phi_enc = MLP(
            input_dim=h_dim + x_dim,
            n_list=[256],
            f_list=["tanh"],
            dropout=dropout
        )
        # g_t, Backward recurrence
        self.rnn_g = nn.LSTM(256, g_dim, num_layers=self.num_rnn_g, batch_first=True)

        # g_t, z_tm1 -> z_t
        # Inference q(z|g,z)
        self.inf = MLP(
            input_dim=z_dim + g_dim,
            n_list=[64, 32],
            f_list=["tanh", "tanh"],
            dropout=dropout
        )
        self.inf_mean = nn.Linear(32, z_dim)
        self.inf_logvar = nn.Linear(32, z_dim)

        # h_t, z_tm1 -> z_t
        # Prior p(z|h,z)
        self.prior = MLP(
            input_dim=z_dim + h_dim,
            n_list=[64, 32],
            f_list=["tanh", "tanh"],
            dropout=dropout
        )
        self.prior_mean = nn.Linear(32, z_dim)
        self.prior_logvar = nn.Linear(32, z_dim)

        # Decoder p(x|z,h)
        self.dec = MLP(
            input_dim=z_dim + h_dim,
            n_list=[256],
            f_list=["tanh"],
            dropout=dropout
        )
        self.dec_mean = nn.Linear(256, x_dim)
        self.dec_logvar = nn.Linear(256, x_dim)

        self.z_mean = torch.zeros((1, 1, self.z_dim))
        self.z_logvar = torch.zeros((1, 1, self.z_dim))
        self.z_mean_prior = None
        self.z_logvar_prior = None


    def sample_norm(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return torch.addcmul(mean, eps, std)


    def gen_x(self, z, h):
        dec_input = torch.cat((z, h), -1)
        dec_input = self.dec(dec_input)
        logvar = self.dec_logvar(dec_input)
        return logvar


    def gen_z(self, h, z_tm1):
        prior_input = torch.cat((z_tm1, h), -1)
        prior_input = self.prior(prior_input)
        mean_prior = self.prior_mean(prior_input)
        logvar_prior = self.prior_logvar(prior_input)
        return mean_prior, logvar_prior


    def inference(self, x, h):
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(x.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(x.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(x.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(x.device)

        enc_input = torch.cat((x, h), -1)
        enc_input = self.phi_enc(enc_input)
        g_inv, _ = self.rnn_g(torch.flip(enc_input, [0]))
        g = torch.flip(g_inv, [0])

        for t in range(seq_len):
            inf_input = torch.cat((g[t, :, :], z_t), -1)
            inf_input = self.inf(inf_input)
            z_mean[t, :, :] = self.inf_mean(inf_input)
            z_logvar[t, :, :] = self.inf_logvar(inf_input)
            z_t = self.sample_norm(z_mean[t, :, :], z_logvar[t, :, :])
            z[t, :, :] = z_t

        return z, z_mean, z_logvar


    def get_h(self, x_mt1):
        x_h = self.phi_x(x_mt1)
        h, _ = self.rnn_h(x_h)
        return h


    def forward(self, x: Tensor):
        _, batch_size, x_dim = x.shape
        x_0 = torch.zeros((1, batch_size, x_dim)).to(x.device)
        x_mt1 = torch.cat((x_0, x[:-1, :, :]), 0)
        h = self.get_h(x_mt1)
        self.z, self.z_mean, self.z_logvar = self.inference(x, h)
        z_0 = torch.zeros((1, batch_size, self.z_dim)).to(x.device)
        z_tm1 = torch.cat((z_0, self.z[:-1, :, :]), 0)
        self.z_mean_prior, self.z_logvar_prior = self.gen_z(h, z_tm1)
        x_logvar = self.gen_x(self.z, h)

        return x_logvar
