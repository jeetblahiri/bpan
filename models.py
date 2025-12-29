# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedEILayer(nn.Module):
    def __init__(self, input_dim, n_exc, n_inh, dt=0.2, use_dales=True):
        """
        input_dim: dimension of input vector
        n_exc: number of excitatory units
        n_inh: number of inhibitory units
        dt: integration step (alpha)
        use_dales: if False, no sign constraints (ablation)
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.dt = dt
        self.use_dales = use_dales

        # Recurrent parameters (pre-softplus)
        self.theta_EE = nn.Parameter(torch.randn(n_exc, n_exc) * 0.05)
        self.theta_EI = nn.Parameter(torch.randn(n_exc, n_inh) * 0.05)
        self.theta_IE = nn.Parameter(torch.randn(n_inh, n_exc) * 0.05)
        self.theta_II = nn.Parameter(torch.randn(n_inh, n_inh) * 0.05)

        # Input projections
        self.W_XE = nn.Linear(input_dim, n_exc, bias=True)
        self.W_XI = nn.Linear(input_dim, n_inh, bias=True)

        # Bias / thresholds
        self.b_e = nn.Parameter(torch.zeros(n_exc))
        self.b_i = nn.Parameter(torch.zeros(n_inh))

    def get_weights(self):
        """
        Enforce Dale's law AND scale by fan-in so recurrence is stable.
        If use_dales=False, just use unconstrained weights (ablation).
        """
        if not self.use_dales:
            # Unconstrained recurrent weights
            W_EE = self.theta_EE
            W_EI = self.theta_EI
            W_IE = self.theta_IE
            W_II = self.theta_II
            return W_EE, W_EI, W_IE, W_II

        # Dale + fan-in scaling
        base_EE = F.softplus(self.theta_EE)      # >= 0
        base_EI = F.softplus(self.theta_EI)      # >= 0
        base_IE = F.softplus(self.theta_IE)      # >= 0
        base_II = F.softplus(self.theta_II)      # >= 0

        # Scale rows by appropriate fan-in
        W_EE = base_EE / self.n_exc             # E <- E, excitatory
        W_IE = base_IE / self.n_exc             # I <- E, excitatory
        W_EI = -base_EI / self.n_inh            # E <- I, inhibitory
        W_II = -base_II / self.n_inh            # I <- I, inhibitory

        return W_EE, W_EI, W_IE, W_II

    def step(self, x, e, i):
        """
        One Euler step of EI dynamics.
        x: [batch, input_dim]
        e: [batch, n_exc]
        i: [batch, n_inh]
        """
        W_EE, W_EI, W_IE, W_II = self.get_weights()

        r_e = F.relu(e)
        r_i = F.relu(i)

        # Input drive
        drive_e_in = self.W_XE(x)  # [batch, n_exc]
        drive_i_in = self.W_XI(x)  # [batch, n_inh]

        # Recurrent input
        rec_e_from_e = r_e @ W_EE.T      # [batch, n_exc]
        rec_e_from_i = r_i @ W_EI.T      # [batch, n_exc]

        rec_i_from_e = r_e @ W_IE.T      # [batch, n_inh]
        rec_i_from_i = r_i @ W_II.T      # [batch, n_inh]

        # Net inputs
        net_e = rec_e_from_e + rec_e_from_i + drive_e_in + self.b_e
        net_i = rec_i_from_e + rec_i_from_i + drive_i_in + self.b_i

        # Euler integration
        e_next = (1.0 - self.dt) * e + self.dt * net_e
        i_next = (1.0 - self.dt) * i + self.dt * net_i

        # E/I balance terms for regularization
        exc_input_e = rec_e_from_e + drive_e_in
        inh_input_e = rec_e_from_i            # (negative if Dale’s law)
        balance_e = exc_input_e + inh_input_e # want ~0

        exc_input_i = rec_i_from_e + drive_i_in
        inh_input_i = rec_i_from_i
        balance_i = exc_input_i + inh_input_i # want ~0

        return e_next, i_next, r_e, r_i, balance_e, balance_i

    def forward(self, x, T=5):
        """
        Run EI dynamics for T steps.
        Returns final excitatory rate and regularization costs.
        """
        batch_size = x.size(0)
        device = x.device

        e = torch.zeros(batch_size, self.n_exc, device=device)
        i = torch.zeros(batch_size, self.n_inh, device=device)

        act_cost = 0.0
        bal_cost = 0.0

        for _ in range(T):
            e, i, r_e, r_i, balance_e, balance_i = self.step(x, e, i)
            act_cost = act_cost + (r_e.pow(2).mean() + r_i.pow(2).mean())
            bal_cost = bal_cost + (balance_e.pow(2).mean() + balance_i.pow(2).mean())

        # Average over time steps
        act_cost = act_cost / T
        bal_cost = bal_cost / T

        return F.relu(e), act_cost, bal_cost

    def forward_with_stats(self, x, T=5):
        """
        Like forward(), but returns per-time-step stats for dynamics analysis.
        Returns:
          r_e_seq: [batch, T, n_exc]
          r_i_seq: [batch, T, n_inh]
          bal_e_seq: [batch, T, n_exc]
          bal_i_seq: [batch, T, n_inh]
        """
        batch_size = x.size(0)
        device = x.device

        e = torch.zeros(batch_size, self.n_exc, device=device)
        i = torch.zeros(batch_size, self.n_inh, device=device)

        r_e_list = []
        r_i_list = []
        bal_e_list = []
        bal_i_list = []

        for _ in range(T):
            e, i, r_e, r_i, balance_e, balance_i = self.step(x, e, i)
            r_e_list.append(F.relu(e))
            r_i_list.append(F.relu(i))
            bal_e_list.append(balance_e)
            bal_i_list.append(balance_i)

        r_e_seq = torch.stack(r_e_list, dim=1)    # [B, T, n_exc]
        r_i_seq = torch.stack(r_i_list, dim=1)    # [B, T, n_inh]
        bal_e_seq = torch.stack(bal_e_list, dim=1)
        bal_i_seq = torch.stack(bal_i_list, dim=1)

        return r_e_seq, r_i_seq, bal_e_seq, bal_i_seq


class BPANClassifier(nn.Module):
    def __init__(self,
                 input_dim=784,
                 n_classes=10,
                 n_exc=128,
                 n_inh=32,
                 T=6,
                 lambda_act=1e-4,
                 lambda_bal=5e-4,
                 use_dales=True,
                 use_act_reg=True,
                 use_bal_reg=True):
        super().__init__()
        self.ei_layer = BalancedEILayer(input_dim, n_exc, n_inh,
                                        dt=0.2, use_dales=use_dales)
        self.readout = nn.Linear(n_exc, n_classes)
        self.T = T
        self.lambda_act = lambda_act
        self.lambda_bal = lambda_bal
        self.use_act_reg = use_act_reg
        self.use_bal_reg = use_bal_reg

    def forward(self, x, targets=None):
        """
        x: [batch, input_dim]
        targets: optional [batch] labels
        """
        r_e, act_cost, bal_cost = self.ei_layer(x, T=self.T)
        logits = self.readout(r_e)

        out = {
            "logits": logits,
            "act_cost": act_cost,
            "bal_cost": bal_cost
        }

        if targets is not None:
            ce = F.cross_entropy(logits, targets)
            reg = 0.0
            if self.use_act_reg:
                reg = reg + self.lambda_act * act_cost
            if self.use_bal_reg:
                reg = reg + self.lambda_bal * bal_cost
            loss = ce + reg
            out["loss"] = loss
            out["ce_loss"] = ce
            out["reg_loss"] = reg

        return out

    def forward_with_stats(self, x):
        """
        For dynamics analysis, not training.
        """
        r_e_seq, r_i_seq, bal_e_seq, bal_i_seq = self.ei_layer.forward_with_stats(
            x, T=self.T
        )
        # You can also compute logits per time step if needed
        logits_seq = self.readout(r_e_seq)  # [B, T, n_classes]
        return {
            "r_e_seq": r_e_seq,
            "r_i_seq": r_i_seq,
            "bal_e_seq": bal_e_seq,
            "bal_i_seq": bal_i_seq,
            "logits_seq": logits_seq,
        }


class MLPClassifier(nn.Module):
    def __init__(self,
                 input_dim=784,
                 n_classes=10,
                 hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x, targets=None):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)

        out = {"logits": logits}

        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            out["loss"] = loss

        return out


class ConvBackboneCIFAR(nn.Module):
    """
    Small conv net for CIFAR-10 feature extraction.
    Output is flattened feature vector.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16x16 -> 8x8
        )
        self.out_dim = 64 * 8 * 8

    def forward(self, x):
        h = self.features(x)
        return h.view(h.size(0), -1)


class ConvWrapper(nn.Module):
    """
    Wraps a conv backbone with a head (MLP or BPAN).
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        return self.head(feats, targets=targets)

    def forward_with_stats(self, x):
        """
        Only valid if head is a BPANClassifier (uses forward_with_stats).
        """
        feats = self.backbone(x)
        return self.head.forward_with_stats(feats)

