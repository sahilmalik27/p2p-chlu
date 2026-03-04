"""CLI entry point for CHLU experiments.

Usage:
    python -m chlu.cli exp-a [--epochs 100] [--device cpu]
    python -m chlu.cli exp-b [--epochs 100] [--device cpu]
    python -m chlu.cli exp-c [--epochs 50]  [--device cuda]
    python -m chlu.cli all   [--epochs 100] [--device cpu]
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="chlu",
        description="CHLU — Causal Hamiltonian Learning Unit experiments",
    )
    subparsers = parser.add_subparsers(dest="command", help="Experiment to run")

    # Shared arguments
    def add_common_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--epochs", type=int, default=100, help="Training epochs")
        sub.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
        sub.add_argument("--output-dir", type=str, default=None, help="Output directory")

    # Exp A
    p_a = subparsers.add_parser("exp-a", help="Lemniscate stability (50-cycle rollout)")
    add_common_args(p_a)

    # Exp B
    p_b = subparsers.add_parser("exp-b", help="Perturbed sine wave kinetic safety")
    add_common_args(p_b)

    # Exp C
    p_c = subparsers.add_parser("exp-c", help="MNIST generation via Langevin dynamics")
    add_common_args(p_c)
    p_c.add_argument("--n-per-digit", type=int, default=10, help="Samples per digit")

    # All
    p_all = subparsers.add_parser("all", help="Run all experiments")
    add_common_args(p_all)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command in ("exp-a", "all"):
        from chlu.experiments.exp_a_stability import run as run_a

        out = args.output_dir or "results/exp_a"
        run_a(output_dir=out, epochs=args.epochs, device=args.device)

    if args.command in ("exp-b", "all"):
        from chlu.experiments.exp_b_safety import run as run_b

        out = args.output_dir or "results/exp_b"
        run_b(output_dir=out, epochs=args.epochs, device=args.device)

    if args.command in ("exp-c", "all"):
        from chlu.experiments.exp_c_generate import run as run_c

        out = args.output_dir or "results/exp_c"
        n_per = getattr(args, "n_per_digit", 10)
        run_c(output_dir=out, epochs=args.epochs, device=args.device, n_per_digit=n_per)


if __name__ == "__main__":
    main()
