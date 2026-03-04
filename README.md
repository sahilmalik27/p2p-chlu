# CHLU — Causal Hamiltonian Learning Unit

A production-grade PyTorch implementation of the **Causal Hamiltonian Learning Unit** from [arXiv:2603.01768v1](https://arxiv.org/abs/2603.01768) (ICLR 2026).

CHLU is a drop-in replacement for LSTM/Neural ODE in temporal tasks, using relativistic Hamiltonian dynamics with symplectic integration.

## Architecture

```
Input x → Encoder → (q₀, p₀) → Velocity Verlet → (q_T, p_T) → Decoder → Output ŷ
                         ↑
            Relativistic Hamiltonian H(q,p) = T(p) + V_θ(q) + α‖q‖²
```

**Key components:**
- **Relativistic kinetic energy**: T(p) = √(c² pᵀ M⁻¹ p + m₀² c⁴) — velocity bounded by c
- **Learnable potential**: V_θ(q) — MLP-parameterized energy landscape
- **Confinement**: α‖q‖² — prevents state divergence
- **Symplectic integrator**: Dissipative Velocity Verlet (energy-preserving when γ=0)
- **Training**: Hamiltonian Contrastive Divergence (wake-sleep with replay buffer)

## Installation

```bash
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from chlu import CHLUUnit

# Create a CHLU model
model = CHLUUnit(
    input_dim=2,        # Feature dimension
    latent_dim=16,      # Phase space dimension
    c=2.0,              # Speed limit
    dt=0.01,            # Integration step size
    n_steps=10,         # Verlet steps per forward pass
)

# Single-step prediction
x = torch.randn(32, 2)
y_hat = model(x)

# Sequence generation
seq = model.evolve_sequence(x, seq_len=100)  # (32, 100, 2)
```

## Experiments

Reproduce all three paper experiments via CLI:

```bash
# Exp A: Long-horizon stability (lemniscate, 50-cycle rollout)
python -m chlu exp-a --epochs 100

# Exp B: Kinetic safety (perturbed sine, velocity bounding)
python -m chlu exp-b --epochs 100

# Exp C: MNIST generation (Langevin dynamics)
python -m chlu exp-c --epochs 50 --device cuda

# Run all experiments
python -m chlu all --epochs 100
```

Results (plots + metrics) are saved to `results/exp_{a,b,c}/`.

### Experiment A: Long-Horizon Stability

Train on 3 cycles of a lemniscate (figure-8), then roll out 50 cycles. CHLU maintains bounded orbits where LSTM/Neural ODE diverge.

### Experiment B: Kinetic Safety

Train on sine waves with ω ~ U(0.5, 2.0). Perturb initial velocity by 5x at inference. CHLU's relativistic kinetic energy bounds velocity at c — perturbation causes only a phase shift, not amplitude blowup.

### Experiment C: Thermodynamic Generation

Train autoencoder on 10k MNIST images. Generate via Langevin dynamics from class centroids with temperature annealing. Recognizable digit modes emerge.

## Inference Modes

```python
# Conservative (training, energy-preserving)
model.set_gamma(0.0)

# Dissipative annealing (drains to attractors)
model.set_gamma(0.1)

# Langevin generative (stochastic sampling)
from chlu import LangevinSampler
sampler = LangevinSampler(model.hamiltonian, dt=0.01, gamma=0.1, temperature=1.0)
q, p = sampler.sample(q0, p0, n_steps=1000)
```

## Project Structure

```
chlu/
├── core/
│   ├── hamiltonian.py      # H(q,p) = T(p) + V_θ(q) + α‖q‖²
│   ├── integrator.py       # Velocity Verlet (symplectic + dissipative)
│   ├── chlu_unit.py        # Main nn.Module
│   └── langevin.py         # Langevin dynamics sampler
├── training/
│   ├── contrastive.py      # Hamiltonian Contrastive Divergence
│   ├── losses.py           # MSE + Lyapunov regularization
│   └── replay_buffer.py    # Replay buffer for sleep phase
├── baselines/
│   ├── lstm_baseline.py    # LSTM comparison
│   └── node_baseline.py    # Neural ODE comparison
├── experiments/
│   ├── exp_a_stability.py  # Lemniscate 50-cycle rollout
│   ├── exp_b_safety.py     # Perturbed sine wave
│   └── exp_c_generate.py   # MNIST generation
├── data/
│   ├── lemniscate.py       # Figure-8 trajectory generator
│   └── sine_wave.py        # Sine wave dataset
├── utils/
│   ├── plotting.py         # Visualization
│   └── metrics.py          # Evaluation metrics
└── cli.py                  # CLI entry point
```

## Testing

```bash
pytest
```

## Citation

```bibtex
@article{jawahar2026chlu,
  title={Causal Hamiltonian Learning Unit},
  author={Jawahar, Pratik and Pierini, Maurizio},
  journal={arXiv preprint arXiv:2603.01768},
  year={2026}
}
```

## License

MIT
