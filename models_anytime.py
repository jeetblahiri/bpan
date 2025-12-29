# models_anytime.py
"""
Anytime baseline models for comparison with BPAN:
  1. ACT (Adaptive Computation Time) - Graves 2016
  2. PonderNet - Banino et al. 2021
  3. Multi-Exit MLP/CNN with confidence-based early stopping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. ACT - Adaptive Computation Time (Graves, 2016)
# ============================================================================

class ACTCell(nn.Module):
    """
    A single recurrent cell with halting mechanism for ACT.
    Uses an RNN/GRU core with a learned halting unit.
    """
    def __init__(self, input_dim, hidden_dim, halt_epsilon=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.halt_epsilon = halt_epsilon
        
        # Core RNN dynamics
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        
        # Halting probability (scalar per sample)
        self.halt_linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, h_prev):
        """
        x: [B, D] input
        h_prev: [B, H] previous hidden state
        Returns: h_new, halt_prob
        """
        h_new = self.rnn_cell(x, h_prev)
        halt_logit = self.halt_linear(h_new)
        halt_prob = torch.sigmoid(halt_logit).squeeze(-1)  # [B]
        return h_new, halt_prob


class ACTClassifier(nn.Module):
    """
    ACT model for classification.
    Runs recurrent computation until cumulative halt probability >= 1-epsilon,
    or until max_steps is reached.
    
    Training: Uses remainder-weighted combination of outputs at all steps.
    Inference: Can early-exit based on halting.
    """
    def __init__(self, input_dim, n_classes, hidden_dim=256, 
                 max_steps=10, halt_epsilon=0.01, time_penalty=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.halt_epsilon = halt_epsilon
        self.time_penalty = time_penalty  # tau in ACT paper
        
        self.act_cell = ACTCell(input_dim, hidden_dim, halt_epsilon)
        self.readout = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x, targets=None):
        """
        x: [B, D] static input (pondered repeatedly)
        """
        B = x.size(0)
        device = x.device
        
        h = torch.zeros(B, self.hidden_dim, device=device)
        
        # Accumulators
        cumulative_halt = torch.zeros(B, device=device)
        remainders = torch.zeros(B, device=device)
        n_updates = torch.zeros(B, device=device)
        
        # Output accumulator (weighted by halt probs)
        output_accum = torch.zeros(B, self.hidden_dim, device=device)
        
        # Track per-step data for analysis
        h_seq = []
        halt_seq = []
        logits_seq = []
        
        for t in range(self.max_steps):
            h, halt_prob = self.act_cell(x, h)
            h_seq.append(h)
            halt_seq.append(halt_prob)
            logits_seq.append(self.readout(h))
            
            # Samples still computing
            still_running = (cumulative_halt < 1.0 - self.halt_epsilon).float()
            
            # New halt probability (only for running samples)
            new_halt = halt_prob * still_running
            
            # Check if this step would exceed threshold
            would_exceed = (cumulative_halt + new_halt > 1.0 - self.halt_epsilon).float()
            
            # Remainder for samples that halt this step
            remainder = (1.0 - cumulative_halt) * would_exceed * still_running
            
            # Update accumulators
            cumulative_halt = cumulative_halt + new_halt * (1.0 - would_exceed)
            cumulative_halt = cumulative_halt + remainder
            
            # Weight for this step's output
            step_weight = new_halt * (1.0 - would_exceed) + remainder
            
            output_accum = output_accum + step_weight.unsqueeze(1) * h
            remainders = remainders + remainder
            n_updates = n_updates + still_running
        
        # Final weighted output
        logits = self.readout(output_accum)
        
        # Ponder cost = expected number of steps
        ponder_cost = n_updates.mean() + remainders.mean()
        
        out = {
            "logits": logits,
            "ponder_cost": ponder_cost,
            "n_updates": n_updates,
            "h_seq": torch.stack(h_seq, dim=1),  # [B, T, H]
            "halt_seq": torch.stack(halt_seq, dim=1),  # [B, T]
            "logits_seq": torch.stack(logits_seq, dim=1),  # [B, T, C]
        }
        
        if targets is not None:
            ce = F.cross_entropy(logits, targets)
            loss = ce + self.time_penalty * ponder_cost
            out["loss"] = loss
            out["ce_loss"] = ce
            out["reg_loss"] = self.time_penalty * ponder_cost
            
        return out
    
    def forward_anytime(self, x, threshold=0.9):
        """
        Early-exit inference based on cumulative halt probability.
        Returns predictions and steps used per sample.
        """
        B = x.size(0)
        device = x.device
        
        h = torch.zeros(B, self.hidden_dim, device=device)
        cumulative_halt = torch.zeros(B, device=device)
        
        done = torch.zeros(B, dtype=torch.bool, device=device)
        preds = torch.zeros(B, dtype=torch.long, device=device)
        steps_used = torch.zeros(B, dtype=torch.long, device=device)
        
        output_accum = torch.zeros(B, self.hidden_dim, device=device)
        
        for t in range(self.max_steps):
            h, halt_prob = self.act_cell(x, h)
            
            still_running = (~done).float()
            new_halt = halt_prob * still_running
            
            # Update output accumulator
            output_accum = output_accum + new_halt.unsqueeze(1) * h
            cumulative_halt = cumulative_halt + new_halt
            
            # Check for halting
            newly_done = (~done) & (cumulative_halt >= threshold)
            if newly_done.any():
                logits = self.readout(output_accum[newly_done])
                preds[newly_done] = logits.argmax(dim=-1)
                steps_used[newly_done] = t + 1
                done = done | newly_done
            
            if done.all():
                break
        
        # Handle samples that didn't halt
        if (~done).any():
            logits = self.readout(output_accum[~done])
            preds[~done] = logits.argmax(dim=-1)
            steps_used[~done] = self.max_steps
            
        return preds, steps_used


# ============================================================================
# 2. PonderNet (Banino et al., 2021)
# ============================================================================

class PonderNetClassifier(nn.Module):
    """
    PonderNet: learns when to stop pondering via a learned halting distribution.
    Uses KL divergence from a geometric prior to regularize pondering time.
    """
    def __init__(self, input_dim, n_classes, hidden_dim=256,
                 max_steps=10, lambda_p=0.5, beta=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.max_steps = max_steps
        self.lambda_p = lambda_p  # Geometric prior parameter
        self.beta = beta  # KL penalty weight
        
        # Recurrent core
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        
        # Halting probability
        self.halt_linear = nn.Linear(hidden_dim, 1)
        
        # Output head
        self.readout = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x, targets=None):
        """
        x: [B, D] static input
        """
        B = x.size(0)
        device = x.device
        
        h = torch.zeros(B, self.hidden_dim, device=device)
        
        # Collect outputs and halt probabilities
        logits_list = []
        lambda_list = []  # halt probabilities
        h_list = []
        
        for t in range(self.max_steps):
            h = self.rnn_cell(x, h)
            h_list.append(h)
            
            logits_t = self.readout(h)
            logits_list.append(logits_t)
            
            halt_logit = self.halt_linear(h)
            lambda_t = torch.sigmoid(halt_logit).squeeze(-1)  # [B]
            lambda_list.append(lambda_t)
        
        logits_seq = torch.stack(logits_list, dim=1)  # [B, T, C]
        lambda_seq = torch.stack(lambda_list, dim=1)  # [B, T]
        h_seq = torch.stack(h_list, dim=1)  # [B, T, H]
        
        # Compute halting distribution p(n) = lambda_n * prod_{i<n}(1 - lambda_i)
        # Use log-space for numerical stability
        log_lambda = torch.log(lambda_seq + 1e-8)
        log_one_minus_lambda = torch.log(1 - lambda_seq + 1e-8)
        
        # Cumulative sum of log(1-lambda) for survival probability
        cumsum_log_survive = torch.cumsum(
            F.pad(log_one_minus_lambda[:, :-1], (1, 0), value=0), dim=1
        )
        
        # p(n) = exp(log_lambda[n] + cumsum_log_survive[n])
        log_p_n = log_lambda + cumsum_log_survive
        p_n = torch.softmax(log_p_n, dim=1)  # Normalize to proper distribution
        
        # Expected output = sum over n of p(n) * logits[n]
        # [B, T, 1] * [B, T, C] -> [B, C]
        logits = (p_n.unsqueeze(-1) * logits_seq).sum(dim=1)
        
        # Expected number of steps
        steps_range = torch.arange(1, self.max_steps + 1, device=device).float()
        expected_steps = (p_n * steps_range).sum(dim=1)
        
        out = {
            "logits": logits,
            "logits_seq": logits_seq,
            "lambda_seq": lambda_seq,
            "p_n": p_n,
            "expected_steps": expected_steps,
            "h_seq": h_seq,
        }
        
        if targets is not None:
            # Reconstruction loss: expected CE over halting distribution
            ce_per_step = torch.stack([
                F.cross_entropy(logits_seq[:, t], targets, reduction='none')
                for t in range(self.max_steps)
            ], dim=1)  # [B, T]
            
            reconstruction_loss = (p_n * ce_per_step).sum(dim=1).mean()
            
            # KL divergence from geometric prior
            # Geometric prior: p_prior(n) = (1-lambda_p)^(n-1) * lambda_p
            log_prior = torch.zeros(self.max_steps, device=device)
            for n in range(self.max_steps):
                log_prior[n] = n * math.log(1 - self.lambda_p + 1e-8) + math.log(self.lambda_p)
            prior = F.softmax(log_prior, dim=0)
            
            # KL(p_n || prior)
            kl_div = (p_n * (torch.log(p_n + 1e-8) - log_prior.unsqueeze(0))).sum(dim=1).mean()
            
            loss = reconstruction_loss + self.beta * kl_div
            
            out["loss"] = loss
            out["ce_loss"] = reconstruction_loss
            out["kl_loss"] = kl_div
            out["reg_loss"] = self.beta * kl_div
            
        return out
    
    def forward_anytime(self, x, threshold=0.9):
        """
        Early-exit based on cumulative halting probability.
        """
        B = x.size(0)
        device = x.device
        
        h = torch.zeros(B, self.hidden_dim, device=device)
        cumulative_halt = torch.zeros(B, device=device)
        
        done = torch.zeros(B, dtype=torch.bool, device=device)
        preds = torch.zeros(B, dtype=torch.long, device=device)
        steps_used = torch.zeros(B, dtype=torch.long, device=device)
        
        for t in range(self.max_steps):
            h = self.rnn_cell(x, h)
            logits = self.readout(h)
            
            halt_logit = self.halt_linear(h)
            lambda_t = torch.sigmoid(halt_logit).squeeze(-1)
            
            # Samples still running
            still_running = ~done
            
            # Update cumulative halt (for running samples only)
            survival_prob = 1.0 - cumulative_halt
            halt_this_step = survival_prob * lambda_t
            cumulative_halt = cumulative_halt + halt_this_step * still_running.float()
            
            # Check for halting
            newly_done = still_running & (cumulative_halt >= threshold)
            if newly_done.any():
                preds[newly_done] = logits[newly_done].argmax(dim=-1)
                steps_used[newly_done] = t + 1
                done = done | newly_done
            
            if done.all():
                break
        
        # Handle non-halted samples
        if (~done).any():
            preds[~done] = logits[~done].argmax(dim=-1)
            steps_used[~done] = self.max_steps
            
        return preds, steps_used


# ============================================================================
# 3. Multi-Exit MLP/CNN with learned early-exit classifiers
# ============================================================================

class MultiExitMLP(nn.Module):
    """
    Multi-exit MLP: multiple classification heads at different depths.
    Each exit has its own confidence, enabling early stopping.
    """
    def __init__(self, input_dim, n_classes, hidden_dim=256, n_exits=6):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_exits = n_exits
        
        # Shared layers between exits
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Blocks between exits
        self.blocks = nn.ModuleList()
        for i in range(n_exits):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ))
        
        # Exit heads
        self.exit_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for _ in range(n_exits)
        ])
        
    def forward(self, x, targets=None):
        """
        x: [B, D]
        """
        h = F.relu(self.input_proj(x))
        
        logits_list = []
        h_list = []
        
        for i in range(self.n_exits):
            h = self.blocks[i](h) + h  # Residual connection
            h_list.append(h)
            logits_i = self.exit_heads[i](h)
            logits_list.append(logits_i)
        
        logits_seq = torch.stack(logits_list, dim=1)  # [B, n_exits, C]
        h_seq = torch.stack(h_list, dim=1)  # [B, n_exits, H]
        
        # Final output = last exit
        logits = logits_seq[:, -1, :]
        
        out = {
            "logits": logits,
            "logits_seq": logits_seq,
            "h_seq": h_seq,
        }
        
        if targets is not None:
            # Train all exits with weighted loss (later exits weighted more)
            weights = torch.linspace(0.5, 1.0, self.n_exits, device=x.device)
            weights = weights / weights.sum()
            
            ce_list = []
            for i in range(self.n_exits):
                ce_i = F.cross_entropy(logits_seq[:, i], targets)
                ce_list.append(ce_i)
            
            ce_weighted = sum(w * ce for w, ce in zip(weights, ce_list))
            
            out["loss"] = ce_weighted
            out["ce_loss"] = ce_list[-1]  # Report final exit CE
            
        return out
    
    def forward_anytime(self, x, threshold=0.9):
        """
        Early-exit based on softmax confidence.
        """
        B = x.size(0)
        device = x.device
        
        h = F.relu(self.input_proj(x))
        
        done = torch.zeros(B, dtype=torch.bool, device=device)
        preds = torch.zeros(B, dtype=torch.long, device=device)
        steps_used = torch.zeros(B, dtype=torch.long, device=device)
        
        for i in range(self.n_exits):
            h = self.blocks[i](h) + h
            logits = self.exit_heads[i](h)
            
            probs = F.softmax(logits, dim=-1)
            conf, cls = probs.max(dim=-1)
            
            newly_done = (~done) & (conf >= threshold)
            preds[newly_done] = cls[newly_done]
            steps_used[newly_done] = i + 1
            done = done | newly_done
            
            if done.all():
                break
        
        # Handle non-exited samples
        if (~done).any():
            preds[~done] = cls[~done]
            steps_used[~done] = self.n_exits
            
        return preds, steps_used


class MultiExitCNN(nn.Module):
    """
    Multi-exit CNN for CIFAR-10: early exit after each conv block.
    Has 4 conv blocks, so n_exits is clamped to max 4.
    """
    def __init__(self, n_classes=10, n_exits=4):
        super().__init__()
        self.n_classes = n_classes
        # CNN has exactly 4 blocks
        self.n_exits = min(n_exits, 4)
        
        # Conv blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 32 -> 16
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 16 -> 8
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8 -> 4
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),  # -> 1x1
            ),
        ])
        
        # Exit heads - after AdaptiveAvgPool2d(1), dims are just the channel counts
        self.exit_channels = [32, 64, 128, 256]
        self.exit_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.exit_channels[i], 128),
                nn.ReLU(),
                nn.Linear(128, n_classes),
            )
            for i in range(self.n_exits)  # Use clamped value
        ])
        
    def forward(self, x, targets=None):
        """
        x: [B, 3, 32, 32]
        """
        h = x
        logits_list = []
        
        for i in range(self.n_exits):
            h = self.blocks[i](h)
            # Exit head handles pooling internally
            logits_i = self.exit_heads[i](h)
            logits_list.append(logits_i)
        
        logits_seq = torch.stack(logits_list, dim=1)
        logits = logits_seq[:, -1, :]
        
        out = {
            "logits": logits,
            "logits_seq": logits_seq,
        }
        
        if targets is not None:
            weights = torch.linspace(0.5, 1.0, self.n_exits, device=x.device)
            weights = weights / weights.sum()
            
            ce_list = [F.cross_entropy(logits_seq[:, i], targets) for i in range(self.n_exits)]
            ce_weighted = sum(w * ce for w, ce in zip(weights, ce_list))
            
            out["loss"] = ce_weighted
            out["ce_loss"] = ce_list[-1]
            
        return out
    
    def forward_anytime(self, x, threshold=0.9):
        B = x.size(0)
        device = x.device
        h = x
        
        done = torch.zeros(B, dtype=torch.bool, device=device)
        preds = torch.zeros(B, dtype=torch.long, device=device)
        steps_used = torch.zeros(B, dtype=torch.long, device=device)
        
        for i in range(self.n_exits):
            h = self.blocks[i](h)
            logits = self.exit_heads[i](h)
            
            probs = F.softmax(logits, dim=-1)
            conf, cls = probs.max(dim=-1)
            
            newly_done = (~done) & (conf >= threshold)
            preds[newly_done] = cls[newly_done]
            steps_used[newly_done] = i + 1
            done = done | newly_done
            
            if done.all():
                break
        
        if (~done).any():
            preds[~done] = cls[~done]
            steps_used[~done] = self.n_exits
            
        return preds, steps_used


# ============================================================================
# Helper: Unified anytime evaluation
# ============================================================================

def evaluate_anytime_model(model, loader, device, thresholds, max_steps, 
                           dataset_name="mnist", is_cifar=False):
    """
    Unified anytime evaluation for ACT, PonderNet, Multi-Exit, and BPAN.
    
    Returns: dict[threshold] -> {acc, avg_steps, step_hist}
    """
    model.eval()
    
    results = {thr: {"correct": 0, "total": 0, "steps_sum": 0,
                     "step_hist": {t: 0 for t in range(1, max_steps + 1)}}
               for thr in thresholds}
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            if dataset_name == "emnist_letters":
                y = y - 1
            
            if not is_cifar:
                x = x.view(x.size(0), -1)
            
            for thr in thresholds:
                preds, steps_used = model.forward_anytime(x, threshold=thr)
                
                correct = (preds == y).sum().item()
                results[thr]["correct"] += correct
                results[thr]["total"] += x.size(0)
                results[thr]["steps_sum"] += steps_used.sum().item()
                
                for t in range(1, max_steps + 1):
                    results[thr]["step_hist"][t] += (steps_used == t).sum().item()
    
    final = {}
    for thr, r in results.items():
        total = r["total"]
        final[thr] = {
            "acc": r["correct"] / total,
            "avg_steps": r["steps_sum"] / total,
            "step_hist": r["step_hist"],
        }
    return final
