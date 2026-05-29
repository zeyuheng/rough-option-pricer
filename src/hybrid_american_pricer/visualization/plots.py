from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_exercise_boundary(boundaries: list[tuple[int, float]], output: Path) -> None:
    if not boundaries:
        raise ValueError("no exercise boundaries to plot")
    data = pd.DataFrame(boundaries, columns=["step", "boundary"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["step"], data["boundary"], marker="o")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Exercise boundary")
    ax.set_title("Estimated early exercise boundary")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_hurst_sensitivity(data: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["hurst"], data["price"], marker="o")
    ax.set_xlabel("Hurst exponent")
    ax.set_ylabel("Option price")
    ax.set_title("Roughness sensitivity")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

