from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SystemParams:
    """Parameters for the two-player ODE system."""
    r: float = 0.01
    p: float = 0.3
    q: float = 0.3
    h1: float = 0.0
    h2: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    z1_0: float = 0.0
    z2_0: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "SystemParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

_FIXED_PARAMS = {"r": 0.01, "p": 0.3, "q": 0.3}
_FIELD_ORDER = ["h1", "h2", "s1", "s2", "a1", "a2", "b1", "b2", "z1_0", "z2_0"]


def _parse_line(line: str) -> SystemParams:
    """Parse one '|'-delimited data line into a SystemParams instance."""
    parts = line.strip().split("|")
    values = [float(p.strip()) for p in parts]

    if len(values) < len(_FIELD_ORDER):
        raise ValueError(
            f"Expected {len(_FIELD_ORDER)} values, got {len(values)}: {line!r}"
        )

    params_dict = _FIXED_PARAMS | dict(zip(_FIELD_ORDER, values))
    return SystemParams.from_dict(params_dict)


def _read_data_lines(path: Path) -> list[str]:
    """Return non-comment, non-empty lines from *path*."""
    lines = []
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def load_params(path: Path, line_index: int = 0) -> SystemParams:
    """Load parameters from *line_index* (0-based) of *path*."""
    lines = _read_data_lines(path)
    if line_index >= len(lines):
        raise IndexError(
            f"Line index {line_index} out of range (file has {len(lines)} data lines)."
        )
    return _parse_line(lines[line_index])


def file_summary(path: Path, preview_count: int = 5) -> None:
    """Print a short preview of the data file."""
    lines = _read_data_lines(path)
    print(f"File '{path}' — {len(lines)} data lines.")
    for idx, line in enumerate(lines[:preview_count]):
        try:
            p = _parse_line(line)
            print(f"  [{idx}]  h1={p.h1:.4f}  h2={p.h2:.4f}  z1_0={p.z1_0:.4f}  z2_0={p.z2_0:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{idx}]  ERROR: {exc}  — {line[:80]!r}")
    if len(lines) > preview_count:
        print(f"  … and {len(lines) - preview_count} more lines.")


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------

def ode_system(z: list[float], _t: float, params: SystemParams) -> list[float]:
    """
    Right-hand side of the two-player ODE:

        dz1/dt = r*z1 + h1*z1² + (s1 - s2)*z1*z2 * (cross terms) - z1*(q*z1 + p*z2)
        dz2/dt = symmetric counterpart for player 2
    """
    z1, z2 = z

    common = z1 * z2
    cross1 = 1.0 + params.b1 * z1 + params.a1 * z2
    cross2 = 1.0 + params.b2 * z2 + params.a2 * z1
    drain = params.q * z1 + params.p * z2

    dz1 = (
        params.r * z1
        + params.h1 * z1 ** 2
        + params.s1 * common * cross1
        - params.s2 * common * cross2
        - z1 * drain
    )
    dz2 = (
        params.r * z2
        + params.h2 * z2 ** 2
        + params.s2 * common * cross2
        - params.s1 * common * cross1
        - z2 * drain
    )

    if any(not np.isfinite(v) for v in (dz1, dz2)):
        return [0.0, 0.0]

    return [dz1, dz2]


def integrate(
    params: SystemParams,
    t_max: float = 100.0,
    n_steps: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the ODE from (z1_0, z2_0) and return (t, solution)."""
    t = np.linspace(0, t_max, n_steps)
    sol = odeint(ode_system, [params.z1_0, params.z2_0], t, args=(params,))
    return t, sol


# ---------------------------------------------------------------------------
# Phase portrait
# ---------------------------------------------------------------------------

def plot_phase_portrait(
    params: SystemParams,
    x_range: tuple[float, float] = (0, 1.0),
    y_range: tuple[float, float] = (0, 1.0),
    n_grid: int = 12,
    arrow_grid: int = 15,
    t_max: float = 50.0,
    n_steps: int = 2000,
    extra_curves: Optional[list[tuple[np.ndarray, np.ndarray, str, str]]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw the phase portrait (trajectories + direction field) for the system.

    Parameters
    ----------
    extra_curves:
        List of (z1_vals, z2_vals, color, label) tuples to overlay.
    """
    t = np.linspace(0, t_max, n_steps)
    x_vals = np.linspace(*x_range, n_grid)
    y_vals = np.linspace(*y_range, n_grid)

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- trajectories ---
    valid = 0
    for x0 in x_vals:
        for y0 in y_vals:
            try:
                sol = odeint(ode_system, [x0, y0], t, args=(params,))
                z1_s, z2_s = sol[:, 0], sol[:, 1]
                if np.all(np.isfinite(z1_s)) and np.all(np.isfinite(z2_s)):
                    ax.plot(z1_s, z2_s, "b-", linewidth=0.5, alpha=0.7)
                    ax.plot(x0, y0, "ro", markersize=2, alpha=0.5)
                    valid += 1
            except Exception:  # noqa: BLE001
                pass

    print(f"Phase portrait: {valid} trajectories drawn.")

    # --- direction field ---
    X, Y = np.meshgrid(
        np.linspace(*x_range, arrow_grid),
        np.linspace(*y_range, arrow_grid),
    )
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                dz1, dz2 = ode_system([X[i, j], Y[i, j]], 0, params)
                U[i, j], V[i, j] = dz1, dz2
            except Exception:  # noqa: BLE001
                pass

    M = np.hypot(U, V)
    with np.errstate(invalid="ignore", divide="ignore"):
        U_n = np.where(M > 0, U / M, 0.0)
        V_n = np.where(M > 0, V / M, 0.0)

    ax.quiver(X, Y, U_n, V_n, M, alpha=0.6, cmap="viridis", scale=30)

    # --- initial point ---
    ax.plot(
        params.z1_0, params.z2_0, "go", markersize=10,
        label=f"Initial point (z1={params.z1_0:.3f}, z2={params.z2_0:.3f})",
    )

    # --- extra curves ---
    if extra_curves:
        for z1_c, z2_c, color, label in extra_curves:
            ax.plot(z1_c, z2_c, color=color, linewidth=2, label=label)

    ax.set_xlabel("z1", fontsize=12)
    ax.set_ylabel("z2", fontsize=12)
    ax.set_title("Phase Portrait", fontsize=14)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# SVM decision boundary  (J1 = J2)
# ---------------------------------------------------------------------------

def _build_features(z: float, h: float, s: float, a: float, b: float) -> np.ndarray:
    """Return the 25-dimensional feature vector φ(z) for one player."""
    return np.array([
        z,
        h * z,
        s * z,
        b * z,
        a * z,
        h ** 2 * z,
        s ** 2 * z,
        b ** 2 * z,
        a ** 2 * z,
        h * s * z,
        h * b * z,
        h * a * z,
        s * b * z,
        s * a * z,
        a * b * z,
        h*z**2, 
        s*z**2, 
        a*z**2, 
        b*z**2, 
        z**2,   
        h*z**3, 
        s*z**3, 
        a*z**3, 
        b*z**3, 
        z**3
    ])


def compute_J_difference(z1: float, z2: float, params: SystemParams, lambdas: np.ndarray) -> float:
    """Return J1(z1) − J2(z2) = Σ λᵢ · (φᵢ(z1) − φᵢ(z2))."""
    phi1 = _build_features(z1, params.h1, params.s1, params.a1, params.b1)
    phi2 = _build_features(z2, params.h2, params.s2, params.a2, params.b2)
    return float(np.dot(lambdas, phi1 - phi2))


def plot_decision_boundary(
    params: SystemParams,
    lambdas: np.ndarray,
    z_range: tuple[float, float] = (-1.0, 1.0),
    n_points: int = 200,
    save_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot the J1 = J2 decision boundary in (z1, z2) space.

    Returns
    -------
    Z1, Z2, J_diff : meshgrid arrays
    """
    z_lin = np.linspace(*z_range, n_points)
    Z1, Z2 = np.meshgrid(z_lin, z_lin)

    J_diff = np.vectorize(compute_J_difference, excluded=["params", "lambdas"])(
        Z1, Z2, params=params, lambdas=lambdas
    )

    J_min, J_max = J_diff.min(), J_diff.max()
    has_zero = J_min < 0 < J_max

    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_fig = True
    else:
        fig = ax.figure

    if has_zero:
        ax.contourf(Z1, Z2, J_diff, levels=[-1e9, 0, 1e9],
                    colors=["tomato", "steelblue"], alpha=0.3)
        contour = ax.contour(
            Z1, Z2, J_diff,
            levels=[0],
            colors="red",
            linewidths=2,
            label="J1 = J2",
        )

    else:
        region_color = "steelblue" if J_min >= 0 else "tomato"
        label_text = "J1 > J2" if J_min >= 0 else "J1 < J2"
        ax.contourf(Z1, Z2, J_diff, levels=[J_min, J_max],
                    colors=[region_color], alpha=0.3)
        mid = (z_range[0] + z_range[1]) / 2
        ax.text(mid, mid, label_text, fontsize=16, ha="center", va="center")
        print(f"No zero crossing — J_diff ∈ [{J_min:.3f}, {J_max:.3f}]")

    if J_max - J_min > 0.01:
        ax.contour(Z1, Z2, J_diff, levels=np.linspace(J_min, J_max, 11),
                   colors="gray", linewidths=0.5, alpha=0.5)

    J0 = compute_J_difference(params.z1_0, params.z2_0, params, lambdas)
    ax.plot(
        params.z1_0, params.z2_0, "go", markersize=10,
        label=(
            f"J1-J2"
        ),
    )

    ax.set_xlabel("z1", fontsize=12)
    ax.set_ylabel("z2", fontsize=12)
    ax.set_title("Decision boundary J1 = J2 in (z1, z2) space", fontsize=14)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    legend_elements = [
        # Красная линия для J1 = J2
        Line2D([0], [0], color='red', linewidth=2, label='J1 = J2'),
        
        # Зеленая точка для начальной позиции
        Line2D([0], [0], marker='o', color='green', markersize=10, 
            linestyle='None', label=f'Initial point (z1_0={params.z1_0:.3f}, z2_0={params.z2_0:.3f})'),
    ]

    ax.legend(handles=legend_elements, loc="best")
    # ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if created_fig:
        plt.show()
    return Z1, Z2, J_diff


# ---------------------------------------------------------------------------
# Time-series plot
# ---------------------------------------------------------------------------

def plot_trajectory(
    params: SystemParams,
    t_max: float = 100.0,
    n_steps: int = 5000,
) -> None:
    """Plot z1(t), z2(t) and the phase-space trajectory."""
    t, sol = integrate(params, t_max=t_max, n_steps=n_steps)

    fig, (ax_t, ax_ph) = plt.subplots(1, 2, figsize=(12, 4))

    ax_t.plot(t, sol[:, 0], "r-", label="z1(t)")
    ax_t.plot(t, sol[:, 1], "b-", label="z2(t)")
    ax_t.set_xlabel("Time t")
    ax_t.set_ylabel("z")
    ax_t.set_title("Time series")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    ax_ph.plot(sol[:, 0], sol[:, 1], "g-", linewidth=2)
    ax_ph.plot(params.z1_0, params.z2_0, "ro", markersize=8, label="Initial point")
    ax_ph.set_xlabel("z1")
    ax_ph.set_ylabel("z2")
    ax_ph.set_title("Phase trajectory")
    ax_ph.legend()
    ax_ph.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# SVM lambdas
# ---------------------------------------------------------------------------

LAMBDAS = np.array([
    -2.757248,
    -0.584375,
    -0.816353,
    -0.181098,
    -0.559748,
    -0.427462,
    -1.505800,
    0.460905,
    -0.102692,
    -0.403703,
    -0.323300,
    0.388930,
    -0.736412,
    -1.010930,
    -0.684573,
    0.521263,
    0.147172,
    0.483578,
    0.365665,
    -0.073846,
    0.291560,
    0.537558,
    -0.173606,
    1.152286,
    0.277632
])

DEFAULT_PARAMS = SystemParams(
    r=0.01, p=0.3, q=0.3,
    h1=0.493, h2=0.010, s1=0.608, s2=0.527,
    a1=0.795, a2=0.335, b1=0.434, b2=0.390,
    z1_0=0.526, z2_0=0.815,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    file_path = Path("Data/TestData.txt")
    selected_line = 442 # 1 -> 3 в файле, 21 -> 23 в файле...: из номера строки в файле надо вычитать 2.

    # --- load parameters ---
    params = DEFAULT_PARAMS
    try:
        file_summary(file_path)
        params = load_params(file_path, line_index=selected_line)
        print(f"\nLoaded parameters from line {selected_line}:")
        for k, v in params.as_dict().items():
            print(f"  {k}: {v}")
    except FileNotFoundError:
        print(f"File '{file_path}' not found — using default parameters.")
    except Exception:  # noqa: BLE001
        print("Error reading file — using default parameters.")
        traceback.print_exc()

    # --- phase portrait ---
    print("\nBuilding phase portrait …")
    z1_range = (0, 1)
    z2_range = (0, 1)

    z_range = (
        0,
        2,
    )

    fig, ax = plot_phase_portrait(
        params,
        x_range=z1_range,
        y_range=z2_range,
        n_grid=10,
        t_max=50,
        # extra_curves=extra,
    )

    # overlay J1-J2 boundary on the same axes
    plot_decision_boundary(
        params,
        LAMBDAS,
        z_range=z_range,
        n_points=300,
        save_path=Path("Plots/decision_boundary_J1_J2.png"),
        ax=ax,
    )

    plt.show()

    # --- time-series + trajectory ---
    print("\nBuilding trajectory …")
    #plot_trajectory(params, t_max=100.0, n_steps=5000)

    # --- decision boundary ---
    print("\nBuilding J1 = J2 decision boundary …")
    print(f"Player 1:  h1={params.h1:.4f}  s1={params.s1:.4f}  a1={params.a1:.4f}  b1={params.b1:.4f}")
    print(f"Player 2:  h2={params.h2:.4f}  s2={params.s2:.4f}  a2={params.a2:.4f}  b2={params.b2:.4f}")




if __name__ == "__main__":
    main()