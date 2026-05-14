"""
Two-Player Dynamical System Analysis
=====================================
Models the interaction of two players via a coupled ODE system,
supports parameter loading from file, phase portrait / time-series
visualization, and an SVM-style decision boundary plot.

ODE system
----------
dz1/dt = r*z1 + h1*z1² + s1*z1*z2*(1 + b1*z1 + a1*z2)
         - s2*z1*z2*(1 + b2*z2 + a2*z1) - z1*(q*z1 + p*z2)

dz2/dt = r*z2 + h2*z2² + s2*z1*z2*(1 + b2*z2 + a2*z1)
         - s1*z1*z2*(1 + b1*z1 + a1*z2) - z2*(q*z1 + p*z2)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Global parameters (fixed throughout the analysis)
R_DEFAULT: float = 0.01
P_DEFAULT: float = 0.30
Q_DEFAULT: float = 0.30

# Lambdas used for the SVM decision boundary (debug: 2-feature mode)
LAMBDAS_DEBUG = np.array([1.0, 21.0])

# Default parameter set used when no data file is available
DEFAULT_PARAMS: dict = {
    "r": R_DEFAULT,
    "p": P_DEFAULT,
    "q": Q_DEFAULT,
    "h1": 0.05,
    "h2": -0.05,
    "s1": 0.10,
    "s2": 0.08,
    "a1": 0.02,
    "a2": 0.02,
    "b1": 0.01,
    "b2": 0.01,
    "z1_0": 0.50,
    "z2_0": 0.30,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SystemParams:
    """All parameters needed to integrate and analyse the ODE system."""

    # Global parameters
    r: float = R_DEFAULT
    p: float = P_DEFAULT
    q: float = Q_DEFAULT

    # Player 1 parameters
    h1: float = 0.05
    s1: float = 0.10
    a1: float = 0.02
    b1: float = 0.01

    # Player 2 parameters
    h2: float = -0.05
    s2: float = 0.08
    a2: float = 0.02
    b2: float = 0.01

    # Initial conditions
    z1_0: float = 0.50
    z2_0: float = 0.30

    # Integration settings (not loaded from file, set programmatically)
    t_max: float = field(default=50.0, repr=False)
    n_steps: int = field(default=2000, repr=False)

    @property
    def z0(self) -> list[float]:
        return [self.z1_0, self.z2_0]

    @property
    def t_span(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_steps)

    def to_dict(self) -> dict:
        return {
            "r": self.r, "p": self.p, "q": self.q,
            "h1": self.h1, "s1": self.s1, "a1": self.a1, "b1": self.b1,
            "h2": self.h2, "s2": self.s2, "a2": self.a2, "b2": self.b2,
            "z1_0": self.z1_0, "z2_0": self.z2_0,
        }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _parse_pipe_line(line: str) -> dict[str, float]:
    """
    Parse one '|'-separated line into a parameter dict.

    Expected column order:
        h1 | h2 | s1 | s2 | a1 | a2 | b1 | b2 | z1_0 | z2_0
    (r, p, q are global and are NOT expected in the file.)
    """
    col_names = ["h1", "h2", "s1", "s2", "a1", "a2", "b1", "b2", "z1_0", "z2_0"]
    parts = [p.strip() for p in line.split("|")]

    parsed: dict[str, float] = {}
    for name, raw in zip(col_names, parts):
        try:
            parsed[name] = float(raw)
        except (ValueError, TypeError):
            warnings.warn(f"Non-numeric value for '{name}': {raw!r} — using default.")
    return parsed


def load_params(
    filepath: Optional[str | Path] = None,
    line_index: int = 0,
    *,
    r: float = R_DEFAULT,
    p: float = P_DEFAULT,
    q: float = Q_DEFAULT,
    t_max: float = 50.0,
    n_steps: int = 2000,
) -> SystemParams:
    """
    Load a :class:`SystemParams` from a pipe-delimited text file.

    Parameters
    ----------
    filepath:
        Path to the parameter file.  If *None* or the file does not exist,
        ``DEFAULT_PARAMS`` are used.
    line_index:
        Zero-based index of the data line to read (header lines that start
        with '#' are skipped automatically).
    r, p, q:
        Global parameters (always supplied explicitly; not read from file).
    t_max, n_steps:
        Integration window settings.

    Returns
    -------
    SystemParams
    """
    base = SystemParams(r=r, p=p, q=q, t_max=t_max, n_steps=n_steps)

    if filepath is None:
        print("[load_params] No file specified — using DEFAULT_PARAMS.")
        return base

    path = Path(filepath)
    if not path.exists():
        warnings.warn(f"File not found: {path} — using DEFAULT_PARAMS.")
        return base

    data_lines = [
        ln for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    if not data_lines:
        warnings.warn("File contains no data lines — using DEFAULT_PARAMS.")
        return base

    if line_index >= len(data_lines):
        warnings.warn(
            f"line_index={line_index} out of range "
            f"(file has {len(data_lines)} data lines) — using line 0."
        )
        line_index = 0

    parsed = _parse_pipe_line(data_lines[line_index])

    # Overlay parsed values on top of defaults
    for attr, val in parsed.items():
        if hasattr(base, attr):
            object.__setattr__(base, attr, val)

    print(f"[load_params] Loaded line {line_index} from '{path}'.")
    return base


def file_summary(filepath: str | Path) -> None:
    """Print a human-readable summary of a parameter file."""
    path = Path(filepath)
    if not path.exists():
        print(f"[file_summary] File not found: {path}")
        return

    lines = path.read_text().splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    print(f"File : {path.resolve()}")
    print(f"Total lines : {len(lines)}  |  Data lines: {len(data_lines)}")
    print("-" * 60)
    for i, ln in enumerate(data_lines):
        print(f"  [{i:>3}]  {ln}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------

def ode_system(
    z: list[float],
    t: float,  # noqa: ARG001  (required by odeint signature)
    params: SystemParams,
) -> list[float]:
    """
    Right-hand side of the two-player ODE system.

    Parameters
    ----------
    z:
        Current state [z1, z2].
    t:
        Current time (unused explicitly; required by odeint).
    params:
        All model parameters.

    Returns
    -------
    [dz1/dt, dz2/dt]
    """
    z1, z2 = z
    p = params

    # Shared interaction terms
    cross_1 = p.s1 * z1 * z2 * (1.0 + p.b1 * z1 + p.a1 * z2)
    cross_2 = p.s2 * z1 * z2 * (1.0 + p.b2 * z2 + p.a2 * z1)

    # Competition / pressure term
    pressure = p.q * z1 + p.p * z2

    dz1 = (
        p.r * z1
        + p.h1 * z1 ** 2
        + cross_1
        - cross_2
        - z1 * pressure
    )

    dz2 = (
        p.r * z2
        + p.h2 * z2 ** 2
        + cross_2
        - cross_1
        - z2 * pressure
    )

    return [dz1, dz2]


def integrate(params: SystemParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the ODE system.

    Returns
    -------
    t   : shape (n_steps,)
    z1  : shape (n_steps,)
    z2  : shape (n_steps,)
    """
    t = params.t_span
    sol = odeint(ode_system, params.z0, t, args=(params,))
    return t, sol[:, 0], sol[:, 1]


# ---------------------------------------------------------------------------
# Feature engineering  (SVM / functional approximation)
# ---------------------------------------------------------------------------

def _build_features(z: float) -> np.ndarray:
    """
    Construct the feature vector φ(z) for a scalar state value.

    Debug mode (current):  φ(z) = [z, z²]   →  dim = 2

    Production extension:  Replace the body with a 25-dimensional map, e.g.
        monomials up to degree 5, radial basis functions, Fourier features …
    The rest of the pipeline is dimension-agnostic as long as ``lambdas``
    matches the returned length.

    Parameters
    ----------
    z : scalar state value

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    # --- DEBUG: 2 features ---------------------------------------------------
    return np.array([z, z ** 2], dtype=float)

    # --- PRODUCTION TEMPLATE: 25 features ------------------------------------
    # Uncomment and adapt once the feature design is finalised.
    # features = []
    # for k in range(1, 6):          # polynomial up to degree 5
    #     features.append(z ** k)
    # for k in range(1, 6):          # sin / cos Fourier terms
    #     features.append(np.sin(k * z))
    #     features.append(np.cos(k * z))
    # # ... add more until len(features) == 25
    # return np.array(features[:25], dtype=float)


def _compute_J_diff(
    z1: float,
    z2: float,
    lambdas: np.ndarray,
) -> float:
    """
    Compute J1 - J2 = λ · (φ(z1) - φ(z2)).

    Parameters
    ----------
    z1, z2  : scalar state values
    lambdas : weight vector, must match len(_build_features(·))

    Returns
    -------
    float
    """
    return float(lambdas @ (_build_features(z1) - _build_features(z2)))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_phase_portrait(
    params: SystemParams,
    *,
    z_range: tuple[float, float] = (-0.1, 2.0),
    n_grid: int = 20,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Draw the phase portrait: direction field + integrated trajectory.

    Parameters
    ----------
    params  : system parameters
    z_range : axis limits for both z1 and z2
    n_grid  : number of grid points per axis for the quiver plot
    ax      : existing Axes to draw on (created if None)
    show    : call plt.show() at the end
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    # --- Direction field ---
    z_lo, z_hi = z_range
    Z1g, Z2g = np.meshgrid(
        np.linspace(z_lo, z_hi, n_grid),
        np.linspace(z_lo, z_hi, n_grid),
    )
    DZ1 = np.zeros_like(Z1g)
    DZ2 = np.zeros_like(Z2g)

    for i in range(n_grid):
        for j in range(n_grid):
            dz = ode_system([Z1g[i, j], Z2g[i, j]], 0.0, params)
            DZ1[i, j], DZ2[i, j] = dz

    magnitude = np.hypot(DZ1, DZ2)
    magnitude[magnitude == 0] = 1.0          # avoid division by zero

    ax.quiver(
        Z1g, Z2g,
        DZ1 / magnitude, DZ2 / magnitude,
        magnitude,
        cmap="coolwarm", alpha=0.6,
        label="_nolegend_",
    )

    # --- Integrated trajectory ---
    t, z1, z2 = integrate(params)
    ax.plot(z1, z2, "k-", linewidth=1.5, label="Trajectory")
    ax.plot(z1[0], z2[0], "go", ms=8, label=f"Start ({params.z1_0:.2f}, {params.z2_0:.2f})")
    ax.plot(z1[-1], z2[-1], "rs", ms=8, label="End")

    ax.set_xlabel("z₁", fontsize=12)
    ax.set_ylabel("z₂", fontsize=12)
    ax.set_title("Phase Portrait", fontsize=13)
    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_trajectory(
    params: SystemParams,
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot z1(t) and z2(t) as time series on a single axes.

    Parameters
    ----------
    params : system parameters
    ax     : existing Axes to draw on (created if None)
    show   : call plt.show() at the end
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    t, z1, z2 = integrate(params)

    ax.plot(t, z1, label="z₁(t)", color="steelblue", linewidth=1.8)
    ax.plot(t, z2, label="z₂(t)", color="coral", linewidth=1.8)

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.set_title("Time Series  z₁(t),  z₂(t)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_decision_boundary(
    lambdas: np.ndarray = LAMBDAS_DEBUG,
    *,
    z_range: tuple[float, float] = (0.0, 2.0),
    n_grid: int = 300,
    params: Optional[SystemParams] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot the SVM decision boundary J1 = J2 on the (z1, z2) plane.

    The boundary is the zero level-set of:
        Δ(z1, z2) = λ · (φ(z1) - φ(z2))

    Regions where Δ > 0 are shaded blue  (player 1 leads),
    regions where Δ < 0 are shaded red   (player 2 leads).

    Parameters
    ----------
    lambdas : weight vector (must match dimension of _build_features)
    z_range : axis limits for both dimensions
    n_grid  : resolution of the evaluation grid
    params  : if provided, the integrated trajectory is overlaid
    ax      : existing Axes (created if None)
    show    : call plt.show() at the end
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    z_lo, z_hi = z_range
    z_vals = np.linspace(z_lo, z_hi, n_grid)
    Z1g, Z2g = np.meshgrid(z_vals, z_vals)

    # Vectorised computation of Δ(z1, z2) over the grid
    # _build_features is written for scalars; vmap it manually
    Delta = np.zeros_like(Z1g)
    for i in range(n_grid):
        for j in range(n_grid):
            Delta[i, j] = _compute_J_diff(Z1g[i, j], Z2g[i, j], lambdas)

    # Filled contour background
    ax.contourf(
        Z1g, Z2g, Delta,
        levels=[-1e9, 0, 1e9],
        colors=["#f9c0c0", "#c0d8f9"],
        alpha=0.55,
    )
    # Decision boundary contour line
    ax.contour(Z1g, Z2g, Delta, levels=[0], colors="black", linewidths=2.0)

    # Diagonal reference line  z1 = z2
    diag = np.array([z_lo, z_hi])
    ax.plot(diag, diag, "k--", linewidth=0.8, alpha=0.5, label="z₁ = z₂")

    # Optional: overlay system trajectory
    if params is not None:
        _, z1, z2 = integrate(params)
        ax.plot(z1, z2, "k-", linewidth=1.5, label="Trajectory")
        ax.plot(z1[0], z2[0], "go", ms=8, label="Start")
        ax.plot(z1[-1], z2[-1], "rs", ms=8, label="End")

    # Proxy patches for the legend
    import matplotlib.patches as mpatches
    p1_patch = mpatches.Patch(color="#c0d8f9", alpha=0.8, label="J₁ > J₂  (player 1 leads)")
    p2_patch = mpatches.Patch(color="#f9c0c0", alpha=0.8, label="J₁ < J₂  (player 2 leads)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [p1_patch, p2_patch], fontsize=8)

    ax.set_xlabel("z₁", fontsize=12)
    ax.set_ylabel("z₂", fontsize=12)
    ax.set_title(
        f"Decision Boundary  (J₁ = J₂)\n"
        f"λ = {lambdas}   |   φ(z) = [z, z²]",
        fontsize=12,
    )
    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.grid(True, linestyle="--", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# Main / usage example
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end demonstration of the two-player system analysis pipeline.

    Usage examples
    --------------
    1.  Default parameters (no file):
            python two_player_system.py

    2.  Load from file (line 0 by default):
            Uncomment the ``load_params`` call below and supply a path.

        File format (pipe-separated, '#' lines are comments):
            # h1  | h2   | s1  | s2  | a1   | a2   | b1   | b2   | z1_0 | z2_0
            0.05 | -0.05 | 0.10 | 0.08 | 0.02 | 0.02 | 0.01 | 0.01 | 0.50 | 0.30
    """

    # ------------------------------------------------------------------
    # 1. Load parameters
    # ------------------------------------------------------------------
    # params = load_params("params.txt", line_index=0)   # ← from file
    params = load_params()                               # ← default

    print("\nActive parameters:")
    for k, v in params.to_dict().items():
        print(f"  {k:<6} = {v}")

    # ------------------------------------------------------------------
    # 2. Integrate and print quick summary
    # ------------------------------------------------------------------
    t, z1, z2 = integrate(params)
    print(f"\nIntegration: t ∈ [0, {params.t_max}],  {params.n_steps} steps")
    print(f"  z1: [{z1.min():.4f}, {z1.max():.4f}]   final = {z1[-1]:.4f}")
    print(f"  z2: [{z2.min():.4f}, {z2.max():.4f}]   final = {z2[-1]:.4f}")

    # ------------------------------------------------------------------
    # 3. Composite figure: all three plots side by side
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Two-Player Dynamical System", fontsize=15, fontweight="bold")

    plot_phase_portrait(params, ax=axes[0], show=False)
    plot_trajectory(params, ax=axes[1], show=False)
    plot_decision_boundary(LAMBDAS_DEBUG, params=params, ax=axes[2], show=False)

    plt.tight_layout()
    plt.savefig("two_player_system.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved → two_player_system.png")
    plt.show()


if __name__ == "__main__":
    main()
