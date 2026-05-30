
import numpy as np
from typing import Tuple, List, Callable, Optional


def gray_code_sequence(n: int) -> List[np.ndarray]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [np.array([], dtype=int)]
    if n > 20:
        raise ValueError("Gray code enumeration too large for n > 20")
    seq = []
    for g in range(2 ** n):
        gray = g ^ (g >> 1)
        bits = np.array([(gray >> i) & 1 for i in range(n)], dtype=int)
        seq.append(bits)
    return seq


def enumerate_all_energies(energy_func: Callable, n_spins: int) -> Tuple[np.ndarray, np.ndarray]:
    seq = gray_code_sequence(n_spins)
    configs = np.array([2 * bits - 1 for bits in seq], dtype=int)
    energies = np.array([energy_func(c) for c in configs], dtype=float)
    return configs, energies


def exact_partition_function(energies: np.ndarray, beta: float) -> float:
    if beta < 0:
        raise ValueError("beta must be non-negative")
    e_min = energies.min()
    shifted = -beta * (energies - e_min)

    shifted = np.clip(shifted, -700, 700)
    log_z = -beta * e_min + np.log(np.sum(np.exp(shifted)))
    return float(np.exp(log_z))


def exact_thermal_average(energies: np.ndarray, beta: float,
                          observables: np.ndarray) -> float:
    if observables.shape != energies.shape:
        raise ValueError("observables shape must match energies")
    e_min = energies.min()
    shifted = -beta * (energies - e_min)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    z = weights.sum()
    if z < 1e-300:
        return 0.0
    return float(np.dot(weights, observables) / z)


class MetropolisSampler:

    def __init__(self, n_spins: int, energy_func: Callable,
                 beta: float = 1.0, seed: int = 154):
        if n_spins <= 0:
            raise ValueError("n_spins must be positive")
        self.n_spins = n_spins
        self.energy_func = energy_func
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.state = 2 * self.rng.integers(0, 2, size=n_spins) - 1
        self.e_curr = self.energy_func(self.state)

    def sweep(self) -> Tuple[np.ndarray, float]:
        for i in range(self.n_spins):
            self.state[i] *= -1
            e_new = self.energy_func(self.state)
            delta = e_new - self.e_curr
            if delta <= 0:
                accept = True
            else:
                prob = np.exp(-self.beta * delta)
                prob = min(prob, 1.0)
                accept = self.rng.random() < prob
            if accept:
                self.e_curr = e_new
            else:
                self.state[i] *= -1
        return self.state.copy(), self.e_curr

    def sample(self, n_sweeps: int, burn_in: int = 100,
               thinning: int = 10) -> dict:

        for _ in range(burn_in):
            self.sweep()
        states = []
        energies = []
        for k in range(n_sweeps):
            self.sweep()
            if k % thinning == 0:
                states.append(self.state.copy())
                energies.append(self.e_curr)
        return {
            "states": np.array(states),
            "energies": np.array(energies),
        }


class ParallelTemperingSampler:

    def __init__(self, n_spins: int, energy_func: Callable,
                 betas: np.ndarray, seed: int = 154):
        if n_spins <= 0:
            raise ValueError("n_spins must be positive")
        if len(betas) < 2:
            raise ValueError("need at least 2 temperatures for parallel tempering")
        self.n_spins = n_spins
        self.energy_func = energy_func
        self.betas = np.array(betas, dtype=float)
        self.n_replicas = len(betas)
        self.rng = np.random.default_rng(seed)
        self.states = np.array([
            2 * self.rng.integers(0, 2, size=n_spins) - 1
            for _ in range(self.n_replicas)
        ])
        self.energies = np.array([energy_func(s) for s in self.states])

    def replica_exchange_step(self) -> None:
        for i in range(self.n_replicas - 1):
            j = i + 1
            delta = (self.betas[i] - self.betas[j]) * (self.energies[i] - self.energies[j])
            if delta >= 0 or self.rng.random() < np.exp(min(delta, 0.0)):

                self.states[[i, j], :] = self.states[[j, i], :]
                self.energies[[i, j]] = self.energies[[j, i]]

    def local_update_step(self) -> None:
        for r in range(self.n_replicas):
            sampler = MetropolisSampler(self.n_spins, self.energy_func,
                                        self.betas[r], seed=self.rng.integers(0, 2 ** 31))
            sampler.state = self.states[r].copy()
            sampler.e_curr = self.energies[r]
            sampler.sweep()
            self.states[r] = sampler.state
            self.energies[r] = sampler.e_curr

    def sample(self, n_steps: int, exchange_freq: int = 5) -> dict:
        all_states = []
        all_energies = []
        for step in range(n_steps):
            self.local_update_step()
            if step % exchange_freq == 0:
                self.replica_exchange_step()
            all_states.append(self.states.copy())
            all_energies.append(self.energies.copy())
        return {
            "states": np.array(all_states),
            "energies": np.array(all_energies),
            "betas": self.betas.copy(),
        }


class ConditionalProbabilitySampler:

    def __init__(self, n_spins: int, energy_func: Callable,
                 seed: int = 154):
        self.n_spins = n_spins
        self.energy_func = energy_func
        self.rng = np.random.default_rng(seed)

    def sample_given_partial(self, fixed_spins: dict, n_samples: int = 100,
                             beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        free_indices = [i for i in range(self.n_spins) if i not in fixed_spins]
        n_free = len(free_indices)
        samples = []
        energies = []
        for _ in range(n_samples):
            s = np.zeros(self.n_spins, dtype=int)
            for idx, val in fixed_spins.items():
                s[idx] = val

            s[free_indices] = 2 * self.rng.integers(0, 2, size=n_free) - 1

            for _ in range(50):
                i = free_indices[self.rng.integers(0, n_free)]
                s[i] *= -1
                e_new = self.energy_func(s)
                s[i] *= -1
                e_old = self.energy_func(s)
                delta = e_new - e_old
                if delta <= 0 or self.rng.random() < np.exp(-beta * delta):
                    s[i] *= -1
            samples.append(s.copy())
            energies.append(self.energy_func(s))
        return np.array(samples), np.array(energies)
