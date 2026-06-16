"""
Monte Carlo sampling of gauge field configurations.

Adapted from:
  - 1007_snel-repo_spinal-population-dynamics-paper (spinal trajectory evolution)
  - 1057_Chandan118_Bio-Inspired-Navigation (bio-inspired exploration walks)

Algorithms:
    1. Metropolis-Hastings
    2. Spinal trajectory (Langevin) sampling
    3. Bio-inspired Levy-flight exploration

Key formulas:
    Metropolis acceptance: P_accept = min(1, exp(-Delta S))
    Adaptive step: epsilon_new = epsilon_old * (P_accept / P_target)^{1/n}
"""

import numpy as np
from constants import LatticeParams, su2_random, TAU
from lattice_geometry import LatticeGeometry, SiteIndex
from gauge_field import GaugeField, _project_to_su2
import math


class MetropolisSampler:
    def __init__(self, gf: GaugeField, proposal_eps: float = 0.5, seed: int = 0):
        self.gf = gf
        self.p = gf.p
        self.geo = gf.geo
        self.rng = np.random.default_rng(seed)
        if proposal_eps <= 0.0:
            raise ValueError(f"proposal_eps must be > 0, got {proposal_eps}")
        self.eps = proposal_eps
        self.stats = {'proposed': 0, 'accepted': 0, 'action_history': []}

    def single_link_update(self, s: SiteIndex, mu: int) -> bool:
        U_old = self.gf.link(s, mu).copy()
        staple = self.gf.staple_sum(s, mu)
        S_old = -0.5 * self.p.beta * (U_old @ staple).trace().real
        R = _random_su2_perturbation(self.eps, self.rng)
        U_new = _project_to_su2(R @ U_old)
        S_new = -0.5 * self.p.beta * (U_new @ staple).trace().real
        dS = S_new - S_old
        self.stats['proposed'] += 1
        if dS <= 0.0 or self.rng.random() < math.exp(-min(dS, 500.0)):
            self.gf.set_link(s, mu, U_new)
            self.stats['accepted'] += 1
            return True
        return False

    def sweep(self) -> dict:
        accept_count = 0
        total = 0
        for s in self.geo.all_sites():
            for mu in range(4):
                if self.single_link_update(s, mu):
                    accept_count += 1
                total += 1
        S = self.gf.action_density()
        self.stats['action_history'].append(S)
        return {'acceptance_rate': accept_count / max(total, 1), 'action': S}

    def thermalize(self, n_sweeps: int) -> dict:
        if n_sweeps < 1:
            raise ValueError(f"n_sweeps must be >= 1, got {n_sweeps}")
        accept_rates = []
        for _ in range(n_sweeps):
            result = self.sweep()
            accept_rates.append(result['acceptance_rate'])
        return {
            'mean_acceptance': float(np.mean(accept_rates)),
            'final_action': self.stats['action_history'][-1] if self.stats['action_history'] else None,
        }


class SpinalTrajectorySampler:
    """Sample along spinal trajectories in configuration space (Langevin dynamics)."""

    def __init__(self, gf: GaugeField, dtau: float = 0.01, seed: int = 0):
        self.gf = gf
        self.p = gf.p
        self.geo = gf.geo
        self.rng = np.random.default_rng(seed)
        if dtau <= 0.0 or dtau >= 1.0:
            raise ValueError(f"dtau must be in (0, 1), got {dtau}")
        self.dtau = dtau
        self.stats = {'action_history': [], 'step': 0}

    def _action_gradient_at_link(self, s: SiteIndex, mu: int) -> np.ndarray:
        U = self.gf.link(s, mu)
        staple = self.gf.staple_sum(s, mu)
        W = U @ staple
        W_anti = 0.5 * (W - W.conj().T)
        tr_W_anti = np.trace(W_anti)
        grad = W_anti - 0.5 * tr_W_anti * np.eye(2, dtype=complex)
        return -0.5 * self.p.beta * grad

    def step(self) -> float:
        for s in self.geo.all_sites():
            for mu in range(4):
                grad = self._action_gradient_at_link(s, mu)
                xi_a = self.rng.standard_normal(3) * np.sqrt(2.0 * self.dtau)
                xi = 1j * sum(xi_a[a] * TAU[a] for a in range(3))
                drift = -grad * self.dtau + xi
                exp_drift = _su2_exp(drift)
                U_new = _project_to_su2(exp_drift @ self.gf.link(s, mu))
                self.gf.set_link(s, mu, U_new)
        S = self.gf.action_density()
        self.stats['action_history'].append(S)
        self.stats['step'] += 1
        return S

    def run(self, n_steps: int) -> dict:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        for _ in range(n_steps):
            self.step()
        return {
            'final_action': self.stats['action_history'][-1],
            'n_steps': self.stats['step'],
        }


class BioInspiredSampler:
    """Bio-inspired Levy-flight exploration of gauge configuration space."""

    def __init__(self, gf: GaugeField, base_eps: float = 0.3,
                  levy_alpha: float = 1.5, seed: int = 0):
        self.gf = gf
        self.p = gf.p
        self.geo = gf.geo
        self.rng = np.random.default_rng(seed)
        if base_eps <= 0.0:
            raise ValueError(f"base_eps must be > 0, got {base_eps}")
        if not (0.0 < levy_alpha < 2.0):
            raise ValueError(f"levy_alpha must be in (0, 2), got {levy_alpha}")
        self.base_eps = base_eps
        self.levy_alpha = levy_alpha
        self.target_accept = 0.6
        self.current_eps = base_eps
        self.stats = {'accept_history': [], 'action_history': []}

    def _levy_step_size(self) -> float:
        u = self.rng.standard_normal()
        v = self.rng.standard_normal()
        sigma_alpha = (
            (math.gamma(1 + self.levy_alpha) * math.sin(math.pi * self.levy_alpha / 2))
            / (math.gamma(1 + self.levy_alpha / 2) * self.levy_alpha * 2 ** ((self.levy_alpha - 1) / 2))
        ) ** (1.0 / self.levy_alpha)
        step = sigma_alpha * u / (abs(v) ** (1.0 / self.levy_alpha) + 1e-30)
        return abs(step) * self.current_eps

    def sweep(self) -> dict:
        accept = 0
        total = 0
        for s in self.geo.all_sites():
            for mu in range(4):
                eps = max(0.01, min(self._levy_step_size(), 2.0))
                accepted = self._try_update(s, mu, eps)
                if accepted:
                    accept += 1
                total += 1
        rate = accept / max(total, 1)
        self.stats['accept_history'].append(rate)
        self.current_eps *= (rate / self.target_accept) ** 0.1
        self.current_eps = max(0.01, min(self.current_eps, 2.0))
        S = self.gf.action_density()
        self.stats['action_history'].append(S)
        return {'acceptance_rate': rate, 'action': S, 'current_eps': self.current_eps}

    def _try_update(self, s: SiteIndex, mu: int, eps: float) -> bool:
        U_old = self.gf.link(s, mu).copy()
        staple = self.gf.staple_sum(s, mu)
        S_old = -0.5 * self.p.beta * (U_old @ staple).trace().real
        R = _random_su2_perturbation(eps, self.rng)
        U_new = _project_to_su2(R @ U_old)
        S_new = -0.5 * self.p.beta * (U_new @ staple).trace().real
        dS = S_new - S_old
        if dS <= 0.0 or self.rng.random() < math.exp(-min(dS, 500.0)):
            self.gf.set_link(s, mu, U_new)
            return True
        return False

    def run(self, n_sweeps: int) -> dict:
        if n_sweeps < 1:
            raise ValueError(f"n_sweeps must be >= 1, got {n_sweeps}")
        for _ in range(n_sweeps):
            self.sweep()
        return {
            'final_action': self.stats['action_history'][-1],
            'mean_acceptance': float(np.mean(self.stats['accept_history'])),
            'final_eps': self.current_eps,
        }


def _random_su2_perturbation(eps: float, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal(3)
    norm = np.linalg.norm(a) + 1e-30
    n = a / norm
    theta = eps * norm
    n_tau = sum(n[i] * TAU[i] for i in range(3))
    R = math.cos(theta) * np.eye(2, dtype=complex) + 1j * math.sin(theta) * 2.0 * n_tau
    return _project_to_su2(R)


def _su2_exp(A: np.ndarray) -> np.ndarray:
    a3 = A[0, 0].imag
    a1 = -A[0, 1].imag
    a2 = -A[0, 1].real
    alpha = np.sqrt(a1**2 + a2**2 + a3**2 + 1e-30)
    n1, n2, n3 = a1 / alpha, a2 / alpha, a3 / alpha
    n_tau = sum([n1, n2, n3][i] * TAU[i] for i in range(3))
    return np.cos(alpha) * np.eye(2, dtype=complex) + 2j * np.sin(alpha) * n_tau
