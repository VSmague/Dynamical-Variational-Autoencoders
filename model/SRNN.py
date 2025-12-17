import torch
import torch.nn as nn


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
        self.rnn_h = nn.LSTM(256, h_dim, num_layers=self.num_rnn_h)

        # h_t, x_t -> g_t
        self.phi_enc = MLP(
            input_dim=h_dim + x_dim,
            n_list=[256],
            f_list=["tanh"],
            dropout=dropout
        )
        # g_t, Backward recurrence
        self.rnn_g = nn.LSTM(256, g_dim, num_layers=self.num_rnn_g)

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


    def get_h(self, x_mt1):
        x_h = self.phi_x(x_mt1)
        h, _ = self.rnn_h(x_h)
        return h


    def forward(self, x, teaching_forcing_ratio=1.0):
        """
        x : (seq_len, batch, x_dim)
        teaching_forcing_ratio : float between 0 and 1
            Probability of using ground truth x_t as input at each time step.
        Returns:
            x_recon : reconstructed x (seq_len, batch, x_dim)
            recon_loss : reconstruction loss (scalar)
            kld_loss : KL divergence loss (scalar)
        """
        seq_len, batch_size, x_dim = x.size()
        device = x.device

        # ==========================================
        # 1. Pre-calculate Recurrent States (h and g)
        # ==========================================

        # --- A. Forward RNN (h_t depends on x_{t-1}) ---
        x_0 = torch.zeros((1, batch_size, x_dim), device=device)
        x_mt1 = torch.cat((x_0, x[:-1]), dim=0) # Shifted input

        # Calculate h for the whole sequence at once
        # (Assuming self.get_h returns (seq_len, batch, h_dim))
        h_seq = self.get_h(x_mt1) 

        # --- B. Backward RNN (g_t depends on h_t and x_t) ---
        # Concatenate h and x for backward input
        enc_input = torch.cat((h_seq, x), dim=-1)
        
        # MLP feature extractor for backward path
        enc_embed = self.phi_enc(enc_input)
        
        # Run Backward LSTM (Flip -> LSTM -> Flip back)
        g_inv, _ = self.rnn_g(torch.flip(enc_embed, [0]))
        g_seq = torch.flip(g_inv, [0]) # g sequence (seq_len, batch, g_dim)


        # ==========================================
        # 2. Sequential Loop (Latent Sampling)
        # ==========================================
        kld_loss = 0
        recon_loss = 0
        all_recon = []

        # Initial latent state z_{-1} (zeros)
        z_prev = torch.zeros((batch_size, self.z_dim), device=device)
        prev_x = x[0]

        for t in range(seq_len):
            # Extract current step data
            if t == 0 or torch.rand(1).item() < teaching_forcing_ratio:
                x_t = x[t]
            else:
                x_t = prev_x
            
            h_t = h_seq[t]
            g_t = g_seq[t]

            # --- Prior p(z_t | h_t, z_{t-1}) ---
            # Depends on Forward State h and Previous Latent z
            prior_in = torch.cat([z_prev, h_t], dim=1)
            prior_out = self.prior(prior_in)
            prior_mu = self.prior_mean(prior_out)
            prior_logvar = self.prior_logvar(prior_out)

            # --- Inference q(z_t | g_t, z_{t-1}) ---
            # Depends on Backward State g and Previous Latent z
            inf_in = torch.cat([z_prev, g_t], dim=1)
            inf_out = self.inf(inf_in)
            enc_mu = self.inf_mean(inf_out)
            enc_logvar = self.inf_logvar(inf_out)

            # --- Sampling z_t ---
            std = torch.exp(0.5 * enc_logvar)
            eps = torch.randn_like(std)
            z_t = enc_mu + eps * std

            # --- Decoder p(x_t | z_t, h_t) ---
            dec_in = torch.cat([z_t, h_t], dim=1)
            dec_out = self.dec(dec_in)
            dec_mu = self.dec_mean(dec_out)
            dec_logvar = self.dec_logvar(dec_out)

            prev_x = dec_mu

            all_recon.append(dec_mu)

            # --- Loss Accumulation ---
            # Reconstruction Loss
            recon_loss += 0.5 * torch.sum(
                dec_logvar
                + (x_t - dec_mu)**2 / torch.exp(dec_logvar)
            )

            # KL Divergence
            kld_loss += 0.5 * torch.sum(
                prior_logvar - enc_logvar
                + (torch.exp(enc_logvar) + (enc_mu - prior_mu)**2) / torch.exp(prior_logvar)
                - 1
            )
            
            # Update previous z for next step
            z_prev = z_t

        # Create final reconstruction tensor
        x_recon = torch.stack(all_recon)

        return x_recon, recon_loss, kld_loss


    @torch.no_grad()
    def generate(self, seq_len, batch=1, device="cpu", deterministic=False):
        """
        Generates a sequence of samples using the trained SRNN model.
        """
        # 1. Initialize states
        # LSTM hidden states for h (h_n, c_n) - None defaults to zeros
        h_state = None 
        
        # Initial inputs (z_0 and x_0) assumed to be zeros
        z_prev = torch.zeros(batch, self.z_dim, device=device)
        x_prev = torch.zeros(batch, self.x_dim, device=device)

        x_samples = []

        for t in range(seq_len):
            # ------------------------------------------------
            # 1. Deterministic State Update: h_t = RNN(x_{t-1}, h_{t-1})
            # ------------------------------------------------
            # Reshape x_prev for batch_first LSTM: (Batch, 1, x_dim)
            x_prev_in = x_prev.unsqueeze(1)
            
            # Feature extraction phi_x
            phi_x_t = self.phi_x(x_prev_in)
            
            # Run Forward RNN one step
            # h_out shape: (Batch, 1, h_dim)
            h_out, h_state = self.rnn_h(phi_x_t, h_state)
            h_t = h_out.squeeze(1)

            # ------------------------------------------------
            # 2. Prior Step: p(z_t | h_t, z_{t-1})
            # ------------------------------------------------
            prior_in = torch.cat([z_prev, h_t], dim=1)
            prior_out = self.prior(prior_in)
            prior_mu = self.prior_mean(prior_out)
            prior_logvar = self.prior_logvar(prior_out)

            if deterministic:
                z_t = prior_mu
            else:
                std = torch.exp(0.5 * prior_logvar)
                z_t = prior_mu + torch.randn_like(std) * std

            # ------------------------------------------------
            # 3. Decoder Step: p(x_t | z_t, h_t)
            # ------------------------------------------------
            dec_in = torch.cat([z_t, h_t], dim=1)
            dec_out = self.dec(dec_in)
            dec_mu = self.dec_mean(dec_out) 
            dec_logvar = self.dec_logvar(dec_out)

            if deterministic:
                x_t = dec_mu
            else:
                std_x = torch.exp(0.5 * dec_logvar)
                x_t = dec_mu + torch.randn_like(std_x) * std_x

            x_samples.append(x_t)

            # ------------------------------------------------
            # 4. Prepare for next step
            # ------------------------------------------------
            x_prev = x_t
            z_prev = z_t

        return torch.stack(x_samples).squeeze(1) # (seq_len, x_dim)