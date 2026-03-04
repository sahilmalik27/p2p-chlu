"""CHLU — Causal Hamiltonian Learning Unit.

A relativistic symplectic neural dynamics primitive for temporal modeling,
based on arXiv:2603.01768v1.
"""

from chlu.core.chlu_unit import CHLUUnit
from chlu.core.hamiltonian import RelativisticHamiltonian
from chlu.core.integrator import VelocityVerletIntegrator
from chlu.core.langevin import LangevinSampler

__version__ = "0.1.0"
__all__ = [
    "CHLUUnit",
    "RelativisticHamiltonian",
    "VelocityVerletIntegrator",
    "LangevinSampler",
]
