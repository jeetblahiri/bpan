# Balanced Predictive Attractor Networks (BPAN)

Official PyTorch implementation of **"Balanced Predictive Attractor Networks for Anytime and Sequential Inference"** submitted to *Neurocomputing*.

**Jeet Bandhu Lahiri, Siddharth Panwar**  
*Indian Institute of Technology Mandi*

---

## Overview

BPAN introduces **excitatory-inhibitory (E/I) recurrent classifier heads** that achieve parameter-efficient anytime inference without learned halting policies. By incorporating Dale's law constraints and explicit E/I balance regularization, BPAN enables:

- ✅ **5.3× fewer parameters** than ACT/PonderNet baselines
- ✅ **4× speedup** through intrinsic confidence-based early stopping
- ✅ **Biologically-grounded dynamics** with interpretable E/I balance
- ✅ **Zero training modifications** for anytime inference (no ponder costs or KL regularization)

### Key Results

| Model | Parameters | MNIST Accuracy | Avg Steps (θ=0.9) | Speedup |
|-------|-----------|----------------|-------------------|---------|
| **BPAN** | **152,650** | 97.73% | 1.51 | **3.97×** |
| ACT | 803,083 | 98.18% | 2.34 | 2.56× |
| PonderNet | 803,083 | 98.27% | 5.95 | 1.01× |
| Multi-Exit | 614,204 | 98.14% | 1.11 | 5.41× |

---

## Installation
```bash
# Clone repository
git clone https://github.com/jeetblahiri/bpan.git
cd bpan

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- scikit-learn

---

## Quick Start

### Basic Usage
```python
import torch
from models.bpan import BPAN

# Create BPAN classifier head
model = BPAN(
    input_dim=784,           # Feature dimension
    n_excitatory=128,        # Excitatory units
    n_inhibitory=32,         # Inhibitory units
    n_classes=10,            # Number of classes
    n_steps=6,               # Recurrent steps
    alpha=0.5                # Integration constant
)

# Forward pass (training)
x = torch.randn(32, 784)     # Batch of inputs
y = torch.randint(0, 10, (32,))  # Labels

logits_all_steps = model(x)  # Shape: (batch, n_steps, n_classes)

# Compute loss with E/I balance regularization
loss = model.compute_loss(
    x, y,
    lambda_act=1e-4,         # Activity regularization
    lambda_bal=5e-4          # E/I balance regularization
)

# Anytime inference (testing)
logits, steps_used = model.forward_anytime(
    x, 
    confidence_threshold=0.9  # Early stopping threshold
)
```

### Training on MNIST
```bash
# Train BPAN on MNIST
python train_mnist.py --model bpan --hidden 256 --steps 6 --epochs 15

# Train baseline models for comparison
python train_mnist.py --model act --hidden 256 --steps 6
python train_mnist.py --model pondernet --hidden 256 --steps 6
python train_mnist.py --model multiexit --hidden 256 --steps 6
```

### Anytime Inference Evaluation
```bash
# Evaluate anytime performance at different confidence thresholds
python evaluate_anytime.py \
    --model_path checkpoints/bpan_mnist.pt \
    --threshold 0.9 \
    --dataset mnist
```

### Sequential Tasks
```bash
# Cluttered Translated MNIST (6 glimpses)
python train_sequential.py --task ctmnist --glimpses 6 --epochs 20

# Streaming Classification (8 chunks)
python train_sequential.py --task streaming --chunks 8 --epochs 15
```

---

## Model Architecture

BPAN implements discrete-time E/I dynamics with Dale's law sign constraints:
```
e_{t+1} = (1-α)e_t + α(W_EE·r_e(e_t) + W_EI·r_i(i_t) + W_XE·x + b_e)
i_{t+1} = (1-α)i_t + α(W_IE·r_e(e_t) + W_II·r_i(i_t) + W_XI·x + b_i)
```

**Dale's Law Constraints:**
- W_EE, W_IE ≥ 0 (excitatory output)
- W_EI, W_II ≤ 0 (inhibitory output)

**Loss Function:**
```
L = L_CE + λ_act·L_act + λ_bal·L_bal
```

where:
- `L_CE`: Time-averaged cross-entropy
- `L_act`: L2 activity regularization
- `L_bal`: E/I balance regularization (penalizes net recurrent currents)

---

## Experiments

### 1. Anytime Baseline Comparison

Compare BPAN against ACT, PonderNet, and Multi-Exit networks:
```bash
bash scripts/run_anytime_comparison.sh
```

Generates accuracy-vs-speedup curves and halting time distributions.

### 2. Balance Regularizer Ablation

Analyze the effect of E/I balance regularization:
```bash
python ablation_balance.py --hidden 256 --steps 6
```

Outputs:
- Balance current magnitudes and variance
- Robustness to occlusion/noise
- Convergence dynamics

### 3. Sequential Evidence Integration
```bash
# Cluttered Translated MNIST
python train_sequential.py --task ctmnist --glimpses 6

# Streaming Classification
python train_sequential.py --task streaming --chunks 8
```

### 4. Robustness Evaluation
```bash
python evaluate_robustness.py \
    --model_path checkpoints/bpan_mnist.pt \
    --corruption gaussian --severity 0.5
```

Supported corruptions: `gaussian`, `salt_pepper`, `occlusion`

---


## Results Summary

### Standard Classification (MNIST)

| Hidden Size | BPAN Accuracy | MLP Accuracy | Params (BPAN) |
|-------------|---------------|--------------|---------------|
| 64 | 95.70 ± 0.12% | 96.99 ± 0.09% | 39,050 |
| 128 | 96.82 ± 0.10% | 97.65 ± 0.14% | 76,170 |
| 256 | 97.38 ± 0.12% | 98.01 ± 0.12% | 152,650 |
| 512 | 97.92 ± 0.08% | 98.13 ± 0.02% | 307,850 |

### Sequential Evidence Integration

| Task | BPAN Params | GRU Params | BPAN Acc | GRU Acc |
|------|-------------|------------|----------|---------|
| CT-MNIST | 50,250 | 311,306 (6.2×) | 24.98% | 47.68% |
| Streaming | 42,890 | 275,978 (6.4×) | 79.72% | 96.84% |

### Balance Regularizer Effects

- **25% reduction** in recurrent current magnitude
- **40% reduction** in balance variance
- **6.5 pp improvement** in accuracy under 50% occlusion

---

## Citation

TO be updated


 Panwar**: [siddharth@example.com]

**Issues**: Please use the [GitHub Issues](https://github.com/jeetblahiri/bpan/issues) tracker for bug reports and feature requests.
