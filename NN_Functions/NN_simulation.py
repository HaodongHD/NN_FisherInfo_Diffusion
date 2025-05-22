"""nn_simulation.py
====================
A (mostly) self‑contained toolkit to simulate a recurrent neural network (RNN) with
optionally multiple neuronal populations, evaluate Fisher–information–related
metrics, and probe stimulus–response dynamics with both NumPy and PyTorch.

Author
------
Haodong Qin <hqin@salk.edu>
"""
from __future__ import annotations

###############################################################################
# Standard library imports
###############################################################################

from typing import Tuple, List, Optional

###############################################################################
# Third‑party imports
###############################################################################
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn.functional as F

###############################################################################
# Helper functions – array ↔ matrix conversions & derivatives
###############################################################################

def garr_to_G_arr(g_mat: np.ndarray, nn: "NNSim") -> np.ndarray:
    """Compute *G* from population gain matrix *g*.

    Notes
    -----
    `G_{mn} = g_{mn}^2 * f_n` (no summation over *n*).
    """
    f_vec = nn.f.reshape(1, -1)
    f_mat = np.repeat(f_vec, g_mat.shape[0], axis=0)
    return (g_mat ** 2) * f_mat


def Garr_to_g_arr(G_mat: np.ndarray, nn: "NNSim") -> np.ndarray:
    """Inverse of :func:`garr_to_G_arr`. Returns *g* from *G*."""
    f_vec = nn.f.reshape(1, -1)
    f_mat = np.repeat(f_vec, G_mat.shape[0], axis=0)
    return np.sqrt(G_mat / f_mat)


def arr_to_mat(arr: np.ndarray, num_population: int) -> np.ndarray:
    """Reshape a flat array (*n*²,) to square matrix (*n*, *n*)."""
    return arr.reshape(num_population, num_population)


# ---------------------------------------------------------------------------
# Numerical derivatives (symmetric, first‑ and higher‑order)
# ---------------------------------------------------------------------------

def _symmetric_derivative(
    f_minus: np.ndarray | torch.Tensor,
    f_plus: np.ndarray | torch.Tensor,
    h: float,
) -> np.ndarray | torch.Tensor:
    """First‑order symmetric derivative along *h* for *f(t)* averaged across trials."""

    f_minus_mean = f_minus.mean(axis=0)
    f_plus_mean = f_plus.mean(axis=0)
    return (f_minus_mean - f_plus_mean) / (2.0 * h)


def symmetric_derivative_perJ(
    f_t_minus_traj: np.ndarray,
    f_t_plus_traj: np.ndarray,
    h: float,
) -> np.ndarray:
    """Wrapper for NumPy arrays."""
    return _symmetric_derivative(f_t_minus_traj, f_t_plus_traj, h)  # type: ignore[arg-type]


def symmetric_derivative_perJ_torch(
    f_t_minus_traj: torch.Tensor,
    f_t_plus_traj: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Wrapper for PyTorch tensors."""
    return _symmetric_derivative(f_t_minus_traj, f_t_plus_traj, h)  # type: ignore[arg-type]


def symmetric_derivative_perJ_v2(
    f_t_minus_traj: np.ndarray,
    f_t_plus_traj: np.ndarray,
    f_t_2minus_traj: np.ndarray,
    f_t_2plus_traj: np.ndarray,
    h: float,
) -> np.ndarray:
    """Fifth‑order accurate symmetric derivative using four offset evaluations.

    Implements the standard *O(h⁴)* finite‑difference stencil:
    .. math::
        f'(x) = \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}
    """
    f_minus = f_t_minus_traj.mean(axis=0)
    f_plus = f_t_plus_traj.mean(axis=0)
    f_2minus = f_t_2minus_traj.mean(axis=0)
    f_2plus = f_t_2plus_traj.mean(axis=0)
    return (-f_2plus + 8 * f_plus - 8 * f_minus + f_2minus) / (12.0 * h)

###############################################################################
# Linear regression helpers
###############################################################################

def linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:  # noqa: D401
    """Simple linear model *y = a x + b*."""
    return a * x + b


def get_slope(x_data: np.ndarray, y_data: np.ndarray) -> float:
    """Estimate *exp*(slope) after log‑linear fit of first *n* points."""
    params, _ = curve_fit(linear_model, x_data, y_data)
    return np.exp(params[0])


def get_eig_from_slope(eig_timecourse: np.ndarray) -> float:
    """Extract dominant growth‑rate eigenvalue from time‑series."""
    keep = 5  # first points to fit
    x = np.arange(keep)
    eig_vals: List[float] = []
    for col in range(eig_timecourse.shape[1]):
        y = eig_timecourse[:keep, col]
        if np.all(y > 0):
            eig_vals.append(get_slope(x, np.log(y)))
    return float(np.max(eig_vals)) if eig_vals else 0.0

###############################################################################
# Torch helpers
###############################################################################

def pearson_corr_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Vectorised Pearson correlation along dimension *dim*."""
    xm = x - x.mean(dim=dim, keepdim=True)
    ym = y - y.mean(dim=dim, keepdim=True)
    cov = (xm * ym).mean(dim=dim)
    return cov / (torch.sqrt((xm.pow(2).mean(dim=dim) * ym.pow(2).mean(dim=dim))) + eps)

###############################################################################
# Core class definition
###############################################################################

class NNSim:
    """Recurrent neural network simulator with population structure.

    Parameters
    ----------
    g : np.ndarray
        Population‑level gain matrix (*P × P*).
    isFF : bool
        Flag indicating feed‑forward (``True``) or feedback (``False``) architecture.
    N : int
        Total number of neurons in the network.
    f : np.ndarray
        Fraction of neurons in each population (length *P*). Must sum to 1.
    w : float | np.ndarray
        External weight(s) to the first population.
    theta : float
        Initial stimulus magnitude delivered to the first population.
    std_x : float
        Process (state) noise standard deviation.
    std_obs : float
        Observation noise standard deviation (currently unused).
    T_lim : int
        Number of simulation time steps (>= 2).
    num_traj : int
        Number of Monte‑Carlo trajectories per *J* realisation.
    num_J : int
        Number of independent connectivity matrices to sample.
    """

    # ---------------------------------------------------------------------
    # Construction utilities
    # ---------------------------------------------------------------------

    def __init__(
        self,
        g: np.ndarray,
        isFF: bool,
        N: int,
        f: np.ndarray,
        w: float | np.ndarray,
        theta: float,
        std_x: float,
        std_obs: float,
        T_lim: int,
        num_traj: int,
        num_J: int,
    ) -> None:
        # ---------------------------
        # Public hyper‑parameters
        # ---------------------------
        self.g = g.copy()
        self.isFF = bool(isFF)
        self.N = int(N)
        self.f = f.astype(float)
        self.w = w
        self.theta = float(theta)
        self.std_x = float(std_x)
        self.std_obs = float(std_obs)
        self.T_lim = int(T_lim)
        self.num_traj = int(num_traj)
        self.num_J = int(num_J)

        # ---------------------------
        # Derived attributes
        # ---------------------------
        self.n: np.ndarray = self._compute_population_sizes()
        self.input_initialization: float = 1.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_population_sizes(self) -> np.ndarray:
        """Return integer array with neuron counts per population."""
        n = np.floor(self.N * self.f).astype(int)
        n[-1] = self.N - n[:-1].sum()  # ensure exact total
        return n

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def j_matrix(self) -> np.ndarray:
        """Sample a dense *N × N* connectivity matrix *J*.

        The entries within block (*i*, *j*) follow :math:`\mathcal{N}\left(0,
        g_{ij}/\sqrt{N}\right)`.
        """
        self.n = self._compute_population_sizes()  # refresh in case *f* changed
        J = np.zeros((self.N, self.N))
        row_start = 0
        for i, n_i in enumerate(self.n):
            col_start = 0
            for j, n_j in enumerate(self.n):
                block_std = self.g[i, j] / np.sqrt(self.N)
                J[row_start : row_start + n_i, col_start : col_start + n_j] = np.random.normal(
                    0.0, block_std, size=(n_i, n_j)
                )
                col_start += n_j
            row_start += n_i
        return J

    # ------------------------------------------------------------------
    # Simulation wrappers (NumPy & Torch)
    # ------------------------------------------------------------------

    def _initial_input(self) -> np.ndarray:
        """Construct external drive (*N*, 1) adding *theta* to first population."""
        WI0 = np.zeros((self.N, 1))
        WI0[: self.n[0]] = self.theta
        return WI0

    # ..................................................................
    # Single‑trajectory NumPy simulation
    # ..................................................................

    def simulate_single(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single‑trajectory NumPy simulation.

        Returns
        -------
        x_t : np.ndarray
            Raw membrane potentials ``shape=(N, T_lim-1)``.
        S_t : np.ndarray
            Firing rates (``tanh`` non‑linearity) with same shape.
        """
        J = self.j_matrix()

        x_t = np.zeros((self.N, self.T_lim - 1))
        S_t = np.zeros_like(x_t)

        # Initial condition (t = 0)
        x = np.zeros((self.N, 1))
        S = np.tanh(x + self._initial_input())
        x = J @ S + np.random.normal(0.0, self.std_x, size=(self.N, 1))

        for t in range(self.T_lim - 1):
            S = np.tanh(x)
            S_t[:, t] = S.ravel()
            x = J @ S + np.random.normal(0.0, self.std_x, size=(self.N, 1))
            x_t[:, t] = x.ravel()
        return x_t, S_t

    # ..................................................................
    # Multi‑trajectory NumPy simulation (vectorised)
    # ..................................................................

    def simulate_multi_np(self, J: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate *num_traj* trajectories in parallel (NumPy)."""
        J = self.j_matrix() if J is None else J
        WI0 = np.zeros((self.N, self.num_traj))
        WI0[: self.n[0], :] = self.theta

        traj = np.zeros((self.num_traj, self.N, self.T_lim - 1))
        rates = np.zeros_like(traj)

        x = np.zeros((self.N, self.num_traj))
        S = np.tanh(x + WI0)
        x = J @ S + np.random.normal(0.0, self.std_x, size=(self.N, self.num_traj))

        traj[:, :, 0] = x.T
        rates[:, :, 0] = S.T

        for t in range(1, self.T_lim - 1):
            S = np.tanh(x)
            x = J @ S + np.random.normal(0.0, self.std_x, size=(self.N, self.num_traj))
            traj[:, :, t] = x.T
            rates[:, :, t] = S.T
        return traj, rates

    # ..................................................................
    # Multi‑trajectory PyTorch simulation (GPU‑ready)
    # ..................................................................

    def simulate_multi_torch(
        self, J: Optional[torch.Tensor] = None, device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accelerated multi‑trajectory simulation using PyTorch."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        J_t = (
            torch.from_numpy(self.j_matrix()) if J is None else J
        ).to(dtype=torch.float32, device=device)

        WI0 = torch.zeros((self.N, self.num_traj), dtype=torch.float32, device=device)
        WI0[: self.n[0], :] = self.theta

        traj = torch.zeros((self.num_traj, self.N, self.T_lim - 1), device=device)
        rates = torch.zeros_like(traj)

        x = torch.zeros((self.N, self.num_traj), dtype=torch.float32, device=device)
        S = torch.tanh(x + WI0)
        x = J_t @ S + self.std_x * torch.randn_like(x)

        traj[:, :, 0] = x.T
        rates[:, :, 0] = S.T

        for t in range(1, self.T_lim - 1):
            S = torch.tanh(x)
            x = J_t @ S + self.std_x * torch.randn_like(x)
            traj[:, :, t] = x.T
            rates[:, :, t] = S.T
        return traj, rates

    # ------------------------------------------------------------------
    # Fisher‑information utilities (NumPy & Torch)
    # ------------------------------------------------------------------

    def _fisher_three_way_np(
        self, delta_theta: float, use_torch: bool = False
    ) -> Tuple:
        """Return trajectories for θ, θ±Δθ (NumPy or Torch)."""
        theta_orig = self.theta
        J = self.j_matrix()

        # Baseline θ
        traj_θ, S_θ = self.simulate_multi_np(J) if not use_torch else self.simulate_multi_torch(torch.from_numpy(J))

        # θ + Δθ
        self.theta = theta_orig + delta_theta
        traj_p, S_p = self.simulate_multi_np(J) if not use_torch else self.simulate_multi_torch(torch.from_numpy(J))

        # θ - Δθ
        self.theta = theta_orig - delta_theta
        traj_m, S_m = self.simulate_multi_np(J) if not use_torch else self.simulate_multi_torch(torch.from_numpy(J))

        self.theta = theta_orig  # restore
        return traj_θ, S_θ, traj_p, S_p, traj_m, S_m

    # Additional analytic / helper methods can follow …

###############################################################################
# If executed as a script – simple smoke test
###############################################################################

if __name__ == "__main__":
    P = 2  # populations
    g = np.array([[1.5, 0.8], [0.8, 1.2]])
    f = np.array([0.5, 0.5])

    sim = NNSim(
        g=g,
        isFF=False,
        N=400,
        f=f,
        w=1.0,
        theta=0.1,
        std_x=0.05,
        std_obs=0.0,
        T_lim=100,
        num_traj=20,
        num_J=1,
    )

    x_t, S_t = sim.simulate_single()
    print("Simulation finished – x_t shape:", x_t.shape)
