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
        return None   # activation linéaire = pas d'activation
    raise ValueError(f"Activation inconnue: {name}")


class MLP(nn.Module):
    def __init__(self, input_dim, n_list, f_list):
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

            prev_dim = n

        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)


class VRNN(nn.Module):
    def __init__(self, x_dim=80, h_dim=256, z_dim=32, phi_x_dim=32, phi_z_dim=16):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Features extractors phi

        self.phi_x = MLP(
            input_dim=x_dim,
            n_list=[256, phi_x_dim],
            f_list=["tanh", "linear"]
        )

        self.phi_z = MLP(
            input_dim=z_dim,
            n_list=[32, 64, phi_z_dim],
            f_list=["tanh", "tanh", "linear"]
        )

        # Encoder q(z|x,h)
        self.enc = MLP(
            input_dim=phi_x_dim + h_dim,
            n_list=[256],
            f_list=["tanh"]
            )
        self.enc_mu = nn.Linear(256, z_dim)
        self.enc_logvar = nn.Linear(256, z_dim)

        # Prior p(z|h)
        self.prior = MLP(
            input_dim=h_dim,
            n_list=[256],
            f_list=["tanh"]
        )
        self.prior_mu = nn.Linear(256, z_dim)
        self.prior_logvar = nn.Linear(256, z_dim)

        # Decoder p(x|z,h)
        self.dec = MLP(
            input_dim=phi_z_dim + h_dim,
            n_list=[256],
            f_list=["tanh"]
        )
        self.dec_mu = nn.Linear(256, x_dim)
        self.dec_logvar = nn.Linear(256, x_dim)

        # RNN
        self.rnn = nn.GRU(phi_x_dim + phi_z_dim, h_dim)

    def forward(self, x):
        """
        x : (seq_len, batch, x_dim)
        """
        seq_len, batch, _ = x.size()
        h = torch.zeros(1, batch, self.h_dim, device=x.device)

        kld_loss = 0
        recon_loss = 0
        # LISTE POUR STOCKER L'AUDIO RECONSTRUIT
        all_recon = []

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

            #reconstruction x du modèle
            all_recon.append(dec_mu)
            x_recon = torch.stack(all_recon)

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

        return x_recon, recon_loss, kld_loss



    def generate(self, seq_len=200, device='cuda'):
        """
        Génère une séquence audio "à partir de rien" en échantillonnant le prior.
        seq_len : nombre de frames temporelles à générer (ex: 200 frames ~= quelques secondes)
        """
        self.eval() # Mode évaluation (important pour les dropouts si vous en avez)
        
        # 1. Initialisation (Tout à zéro pour commencer)
        # Batch size de 1 car on génère un seul exemple
        h = torch.zeros(1, 1, self.h_dim, device=device)
        
        # On a besoin d'un premier x_t fictif pour amorcer le modèle 
        x_t = torch.zeros(1, self.x_dim, device=device)
        
        generated_frames = []
        
        with torch.no_grad():
            for t in range(seq_len):
                # a. Features de l'input précédent
                phi_x_t = self.phi_x(x_t)
                
                # b. Prior p(z|h) 
                prior_h = self.prior(h.squeeze(0))
                prior_mu = self.prior_mu(prior_h)
                prior_logvar = self.prior_logvar(prior_h)
                
                # c. SAMPLING DANS L'ESPACE LATENT 
                # On tire un z aléatoire basé sur la distribution du prior
                std = torch.exp(0.5 * prior_logvar)
                eps = torch.randn_like(std)
                z_t = prior_mu + eps * std  # <--- C'est ici que la magie opère !
                
                phi_z_t = self.phi_z(z_t)
                
                # d. Decoder p(x|z,h)
                dec_h = self.dec(torch.cat([phi_z_t, h.squeeze(0)], dim=1))
                
                # la sortie audio correspond à la moyenne 
                dec_mu = self.dec_mu(dec_h)
                
                # e. On stocke le frame généré
                generated_frames.append(dec_mu)
                
                # f. MISE À JOUR AUTO-RÉGRESSIVE 
                # La sortie actuelle devient l'entrée du pas suivant
                x_t = dec_mu 
                
                # Update RNN state
                rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0)
                _, h = self.rnn(rnn_input, h)
        
        # On empile la liste en un tenseur final (Seq_len, x_dim)
        return torch.stack(generated_frames).squeeze(1)


    def generate_with_priming(self, seq_len=200, prime_sequence=None, device='cuda',noise_level=0.8):
        """
        prime_sequence : Un vrai tenseur audio (Seq_Prime, x_dim) pour amorcer la mémoire
        """
        self.eval()
        
        # 1. Initialisation
        h = torch.zeros(1, 1, self.h_dim, device=device)
        
        # Initialisation de x_t
        if prime_sequence is not None:
            # On prend le premier frame de l'amorce
            x_t = prime_sequence[0].unsqueeze(0) # (1, x_dim)
        else:
            x_t = torch.zeros(1, self.x_dim, device=device)
            
        generated_frames = []
        
        with torch.no_grad():
            
            # PHASE 1 : AMORÇAGE 
            # Si on a une amorce, on force le modèle à la suivre sans générer
            if prime_sequence is not None:
                prime_len = prime_sequence.size(0)
                for t in range(prime_len - 1): # On s'arrête juste avant la fin
                    x_t_real = prime_sequence[t].unsqueeze(0)
                    
                    # On fait juste tourner le RNN pour mettre à jour 'h'
                    # On n'utilise PAS le sampling ici, on veut juste l'état caché
                    phi_x_t = self.phi_x(x_t_real)
                    
                    # Pour mettre à jour h correctement, on doit quand même estimer un z 
                    # On utilise l'encodeur (comme en training) car on a la vraie donnée !
                    prior_h = self.prior(h.squeeze(0))
                    enc_h = self.enc(torch.cat([phi_x_t, h.squeeze(0)], dim=1))
                    enc_mu = self.enc_mu(enc_h)
                    # On utilise la moyenne de l'encodeur (le z le plus probable)
                    z_t = enc_mu 
                    
                    phi_z_t = self.phi_z(z_t)
                    
                    # Update RNN
                    rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0)
                    _, h = self.rnn(rnn_input, h)
                    
                # Le dernier x_t devient le point de départ de la génération
                x_t = prime_sequence[-1].unsqueeze(0)

            # PHASE 2 : GÉNÉRATION PURE
            for t in range(seq_len):
                phi_x_t = self.phi_x(x_t)
                
                # Prior p(z|h)
                prior_h = self.prior(h.squeeze(0))
                prior_mu = self.prior_mu(prior_h)
                prior_logvar = self.prior_logvar(prior_h)
                
                # Sampling Z (Avec une petite astuce "température")
                std = torch.exp(0.5 * prior_logvar)
                # Astuce : on peut réduire le bruit (ex: 0.8) pour stabiliser
                
                eps = torch.randn_like(std) * noise_level
                z_t = prior_mu + eps * std 
                
                phi_z_t = self.phi_z(z_t)
                
                # Decoder
                dec_h = self.dec(torch.cat([phi_z_t, h.squeeze(0)], dim=1))
                dec_mu = self.dec_mu(dec_h)
                
                generated_frames.append(dec_mu)
                
                # Auto-régressif
                x_t = dec_mu 
                
                rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0)
                _, h = self.rnn(rnn_input, h)
        
        return torch.stack(generated_frames).squeeze(1)
