# PLAN.md — p2p-chlu
## Causal Hamiltonian Learning Unit (CHLU) — Production Implementation

**Paper**: arXiv:2603.01768v1 (ICLR 2026 — AI & PDE)
**Authors**: Pratik Jawahar, Maurizio Pierini
**Reference Code**: https://github.com/PRAkTIKal24/CHLU

---

## Goal

Build a clean, production-grade PyTorch implementation of the CHLU primitive with:
1. The core CHLU module (drop-in replacement for LSTM/Neural ODE in temporal tasks)
2. All 3 paper experiments reproduced
3. A pip-installable package with CLI
4. Benchmarks against LSTM and Neural ODE baselines

---

## Architecture

### Core: CHLU Unit
- **State**: z = (q, p) — generalized positions and momenta
- **Hamiltonian**: H(q,p) = T(p) + V_θ(q) + α‖q‖²
  - T(p) = sqrt(c² pᵀ M⁻¹ p + m₀² c⁴)  — relativistic kinetic energy
  - V_θ(q) — learnable MLP potential
  - α‖q‖² — confinement term
- **Mass matrix M**: learnable diagonal, positive-definite (use exp parameterization)
- **Integration**: Dissipative Velocity Verlet (symplectic)
  1. p_{t+0.5} = p_t - (ε/2) ∇_q [V_θ(q_t) + α‖q_t‖²]
  2. q_{t+1} = q_t + ε ∇_p T(p_{t+0.5})
  3. p*_{t+1} = p_{t+0.5} - (ε/2) ∇_q [V_θ(q_{t+1}) + α‖q_{t+1}‖²]
  4. p_{t+1} = (1 - γ) p*_{t+1}

### Relativistic Velocity (bounded by c):
∇_p T = c² M⁻¹ p / sqrt(c² pᵀ M⁻¹ p + m₀² c⁴)

### Training: Hamiltonian Contrastive Divergence
- **Wake phase**: evolve z₀ with γ=0, compute MSE(z_wake, z_target) + λ L_lyapunov
- **Sleep phase**: sample from replay buffer, evolve with γ=0
- **Update**: Δθ ∝ -∇_θ H(z_wake) + ∇_θ H(z_sleep)
- **Replay buffer**: stores hallucinated states from sleep phase

### Inference Modes:
1. **Conservative** (γ=0, T=0): deterministic, energy-preserving
2. **Dissipative annealing** (γ>0, T=0): drains entropy → nearest attractor
3. **Langevin generative** (γ>0, T>0): dp = -∇V dt - γp dt + sqrt(2γkT) dW

---

## Project Structure

```
p2p-chlu/
├── README.md
├── pyproject.toml
├── chlu/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── hamiltonian.py      # Hamiltonian H(q,p), T(p), V_θ(q)
│   │   ├── integrator.py       # Velocity Verlet (symplectic + dissipative)
│   │   ├── chlu_unit.py        # Main CHLU module (nn.Module)
│   │   └── langevin.py         # Langevin dynamics for generation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── contrastive.py      # Hamiltonian Contrastive Divergence
│   │   ├── losses.py           # MSE + Lyapunov regularization
│   │   └── replay_buffer.py    # Replay buffer for sleep phase
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── lstm_baseline.py
│   │   └── node_baseline.py    # Neural ODE baseline
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── exp_a_stability.py  # Lemniscate tracing (50 cycles)
│   │   ├── exp_b_safety.py     # Perturbed sine wave
│   │   └── exp_c_generate.py   # MNIST generation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── lemniscate.py       # Figure-8 trajectory generation
│   │   └── sine_wave.py        # Sine wave dataset
│   ├── cli.py                  # CLI entry point
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py         # Visualization utils
│       └── metrics.py          # Evaluation metrics
└── tests/
    ├── test_hamiltonian.py
    ├── test_integrator.py
    ├── test_chlu_unit.py
    └── test_training.py
```

---

## Experiments to Reproduce

### Exp A: Long-Horizon Stability (Lemniscate)
- Train on 3 cycles of lemniscate (figure-8)
- Infer 50 cycles
- Compare: CHLU vs LSTM vs Neural ODE
- Success: orbit stays closed and bounded

### Exp B: Kinetic Safety (Perturbed Sine)
- 100 sine trajectories, T=1000, ω ~ U(0.5, 2.0)
- Perturb initial states at inference
- Plot kinetic energy and phase space
- Success: velocity saturates at c, perturbation → phase shift only

### Exp C: Thermodynamic Generation (MNIST)
- Train on 10k MNIST images
- Generate: centroid + noise → Langevin dynamics → tanh at step 1000
- Temperature annealing schedule
- Success: recognizable digit modes emerge

---

## Hyperparameters
Reference the paper's default config. Key ones:
- c (speed limit), m₀ (rest mass), α (confinement), ε (step size)
- γ (friction: 0 for training, >0 for inference)
- λ (Lyapunov regularization weight)
- V_θ architecture: small MLP (2-3 hidden layers, ~64-128 units)

---

## Compute Budget
- All experiments should fit in <8GB VRAM (MNIST is only 10k images, small MLPs)
- Use CPU for Exp A and B (tiny state spaces)
- GPU for Exp C (MNIST generation)

---

## Deliverables
1. `pip install .` works
2. `python -m chlu.cli exp-a` reproduces each experiment
3. Plots matching paper figures
4. README with usage, architecture diagram, results
5. Tests passing

---

## Implementation Order
1. Core: hamiltonian.py, integrator.py, chlu_unit.py
2. Training: losses.py, replay_buffer.py, contrastive.py
3. Baselines: LSTM, Neural ODE
4. Data generators: lemniscate, sine wave
5. Exp A → Exp B → Exp C
6. CLI, README, tests
