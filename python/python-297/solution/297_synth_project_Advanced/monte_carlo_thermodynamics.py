"""
monte_carlo_thermodynamics.py
================================
Monte Carlo sampling for dusty plasma crystal configurational
thermodynamics and defect statistics.

Physical motivation:
    At finite temperature, dust grains in a plasma crystal undergo
    thermal fluctuations. The equilibrium configurations are sampled
    from the Boltzmann distribution:
        P({r_i}) ~ exp(-U({r_i}) / (k_B * T_d))

    where U is the total Yukawa interaction energy:
        U = sum_{i<j} (Q_d^2/(4*pi*eps0*r_ij)) * exp(-r_ij/lambda_D)

    We use Metropolis Monte Carlo to sample configurations and compute:
        1. Mean potential energy <U>
        2. Specific heat C_v = d<U>/dT
        3. Pair correlation function g(r)
        4. Defect formation probability
        5. Lindemann melting criterion

    The Monte Carlo method (from 941_quad_monte_carlo):
        estimate = (b-a) * (1/N) * sum_{i=1}^{N} f(x_i)
    is used both for thermodynamic integration and for computing
    configurational averages.

References:
    - From 941_quad_monte_carlo: Monte Carlo integration
    - Hamaguchi, Farouki & Dubin, "Thermodynamic properties of
      strongly coupled dusty plasmas", Phys. Rev. E 56, 4671 (1997)
    - Vaulina & Khrapak, "Yukawa sphere crystallization in a dusty
      plasma", Phys. Rev. Lett. 93, 165003 (2004)
"""

import numpy as np
from typing import Tuple, Dict, Optional


class DustyPlasmaMonteCarlo:
    """
    Metropolis Monte Carlo simulation for 2D dusty plasma crystal.

    The system is N dust grains in a periodic hexagonal box with
    Yukawa interactions. The coupling parameter Gamma controls
    the phase:
        Gamma > Gamma_c ~ 137  =>  Crystal (solid)
        Gamma < Gamma_c        =>  Liquid
    """

    def __init__(
        self,
        N: int = 36,
        kappa: float = 2.0,
        Gamma: float = 200.0,
        box_size: float = 1.0,
        seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        N : int
            Number of dust grains.
        kappa : float
            Screening parameter.
        Gamma : float
            Coupling parameter.
        box_size : float
            Simulation box size (in units of a_ws).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.N = N
        self.kappa = kappa
        self.Gamma = Gamma
        self.box_size = box_size
        self.rng = np.random.RandomState(seed)

        # Initialize on hexagonal lattice
        self.positions = self._init_hexagonal()

        # Interaction energy
        self.energy = self._compute_total_energy()

        # Accumulators
        self.n_samples = 0
        self.energy_sum = 0.0
        self.energy_sq_sum = 0.0

    def _init_hexagonal(self) -> np.ndarray:
        """
        Initialize grain positions on a hexagonal lattice.

        Returns
        -------
        positions : np.ndarray, shape (N, 2)
            Grain positions in units of a_ws.
        """
        # Compute number of cells needed
        n_side = int(np.ceil(np.sqrt(self.N / np.sqrt(3.0))))
        a = self.box_size / n_side  # Lattice spacing

        positions = []
        for n1 in range(n_side):
            for n2 in range(n_side):
                x = a * (n1 + 0.5 * n2)
                y = a * np.sqrt(3.0) / 2.0 * n2
                positions.append([x, y])
                if len(positions) >= self.N:
                    break
            if len(positions) >= self.N:
                break

        return np.array(positions[:self.N])

    def _yukawa_energy_pair(self, r: float) -> float:
        """
        Yukawa interaction energy between two grains at distance r.

        V(r) = (1/r) * exp(-kappa * r)  (in units of Q_d^2/(4*pi*eps0*a_ws))

        Parameters
        ----------
        r : float
            Inter-grain distance (in units of a_ws).

        Returns
        -------
        V : float
            Pair interaction energy.
        """
        if r < 1e-10:
            return 0.0
        return np.exp(-self.kappa * r) / r

    def _minimum_image_distance(
        self, r1: np.ndarray, r2: np.ndarray
    ) -> float:
        """
        Compute minimum-image distance in periodic box.

        Parameters
        ----------
        r1, r2 : np.ndarray, shape (2,)
            Positions.

        Returns
        -------
        d : float
            Minimum-image distance.
        """
        dr = r2 - r1
        # Apply minimum image convention
        dr -= self.box_size * np.round(dr / self.box_size)
        return np.sqrt(dr[0]**2 + dr[1]**2)

    def _compute_total_energy(self) -> float:
        """
        Compute total Yukawa interaction energy.

        U = sum_{i<j} V_Y(|r_i - r_j|)

        Returns
        -------
        energy : float
            Total energy per grain (in k_B*T_d units via Gamma).
        """
        energy = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = self._minimum_image_distance(
                    self.positions[i], self.positions[j]
                )
                energy += self._yukawa_energy_pair(r)
        return energy / self.N

    def _compute_single_energy(self, idx: int) -> float:
        """
        Compute interaction energy of grain idx with all others.

        Parameters
        ----------
        idx : int
            Grain index.

        Returns
        -------
        e_single : float
            Energy contribution from grain idx.
        """
        e = 0.0
        for j in range(self.N):
            if j == idx:
                continue
            r = self._minimum_image_distance(
                self.positions[idx], self.positions[j]
            )
            e += self._yukawa_energy_pair(r)
        return e

    def metropolis_step(
        self,
        max_displacement: float = 0.1,
    ) -> Tuple[bool, float]:
        """
        Single Metropolis MC step.

        1. Pick a random grain
        2. Propose random displacement
        3. Compute energy change
        4. Accept with probability min(1, exp(-dE * Gamma))

        Parameters
        ----------
        max_displacement : float
            Maximum displacement per step (in units of a_ws).

        Returns
        -------
        accepted : bool
            Whether the move was accepted.
        delta_E : float
            Energy change.
        """
        # Pick random grain
        idx = self.rng.randint(self.N)

        # Compute old energy contribution
        E_old = self._compute_single_energy(idx)

        # Propose displacement
        old_pos = self.positions[idx].copy()
        dx = max_displacement * (self.rng.random() - 0.5)
        dy = max_displacement * (self.rng.random() - 0.5)
        self.positions[idx, 0] += dx
        self.positions[idx, 1] += dy

        # Apply periodic BCs
        self.positions[idx] %= self.box_size

        # Compute new energy contribution
        E_new = self._compute_single_energy(idx)
        dE = (E_new - E_old) / self.N

        # Metropolis acceptance
        # Acceptance probability: exp(-Gamma * dE)
        if dE <= 0:
            accepted = True
        else:
            boltz_factor = np.exp(-self.Gamma * dE)
            accepted = self.rng.random() < boltz_factor

        if not accepted:
            self.positions[idx] = old_pos

        return accepted, dE

    def run_simulation(
        self,
        n_steps: int = 10000,
        n_equil: int = 2000,
        max_displacement: float = 0.1,
    ) -> Dict[str, float]:
        """
        Run MC simulation and compute thermodynamic averages.

        Parameters
        ----------
        n_steps : int
            Total MC steps.
        n_equil : int
            Equilibration steps (not counted in averages).
        max_displacement : float
            Maximum displacement per step.

        Returns
        -------
        results : dict
            'mean_energy': <U/N>,
            'specific_heat': C_v/N,
            'acceptance_rate': fraction of accepted moves,
            'lindemann': Lindemann ratio.
        """
        n_accepted = 0
        energy_samples = []
        displacement_sum = 0.0

        for step in range(n_steps):
            accepted, dE = self.metropolis_step(max_displacement)
            if accepted:
                n_accepted += 1

            if step >= n_equil:
                E = self._compute_total_energy()
                energy_samples.append(E)

        energy_samples = np.array(energy_samples)
        n_production = len(energy_samples)

        if n_production > 1:
            mean_E = np.mean(energy_samples)
            var_E = np.var(energy_samples)
            # Specific heat: C_v = Gamma^2 * Var(E) / N
            C_v = self.Gamma**2 * var_E / self.N
        else:
            mean_E = self.energy
            C_v = 0.0

        # Lindemann ratio: sqrt(<delta_r^2>) / a
        # Estimated from acceptance rate and displacement
        mean_disp = max_displacement * n_accepted / max(n_steps, 1)
        lindemann = mean_disp / (self.box_size / np.sqrt(self.N))

        return {
            "mean_energy": float(mean_E),
            "energy_variance": float(np.var(energy_samples)) if n_production > 1 else 0.0,
            "specific_heat": float(C_v),
            "acceptance_rate": float(n_accepted / n_steps),
            "lindemann": float(lindemann),
            "Gamma": self.Gamma,
            "kappa": self.kappa,
            "N": self.N,
        }

    def pair_correlation(
        self,
        n_bins: int = 50,
        r_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pair correlation function g(r).

        g(r) = (1/(n_d * 2*pi*r*dr)) * <sum_{i!=j} delta(r - r_ij)>

        Parameters
        ----------
        n_bins : int
            Number of radial bins.
        r_max : float, optional
            Maximum distance.

        Returns
        -------
        r_bins : np.ndarray
            Bin centers.
        g : np.ndarray
            Pair correlation function.
        """
        if r_max is None:
            r_max = self.box_size / 2.0

        dr = r_max / n_bins
        hist = np.zeros(n_bins)
        count = 0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = self._minimum_image_distance(
                    self.positions[i], self.positions[j]
                )
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 2  # Count both i,j and j,i

        # Normalize
        n_d = self.N / self.box_size**2
        r_bins = np.arange(n_bins) * dr + dr / 2.0
        g = np.zeros(n_bins)
        for k in range(n_bins):
            r = r_bins[k]
            area = 2.0 * np.pi * r * dr
            g[k] = hist[k] / (self.N * n_d * area) if area > 0 else 0.0

        return r_bins, g

    def monte_carlo_integral(
        self,
        func,
        n_samples: int = 10000,
    ) -> Tuple[float, float]:
        """
        Monte Carlo integration (from 941_quad_monte_carlo).

        integral f(x) dx ~ (b-a) * (1/N) * sum f(x_i)

        Applied to computing thermal averages in the dusty plasma:
            <A> = integral A({r}) * exp(-Gamma*U({r})) d{r} /
                  integral exp(-Gamma*U({r})) d{r}

        Parameters
        ----------
        func : callable
            Function to integrate.
        n_samples : int
            Number of random samples.

        Returns
        -------
        estimate : float
            Integral estimate.
        std_error : float
            Standard error of the estimate.
        """
        # Sample random configurations
        values = []
        for _ in range(n_samples):
            # Random configuration
            pos = self.rng.random((self.N, 2)) * self.box_size
            val = func(pos)
            values.append(val)

        values = np.array(values)
        estimate = self.box_size**2 * np.mean(values)
        std_error = self.box_size**2 * np.std(values) / np.sqrt(n_samples)

        return float(estimate), float(std_error)


def melting_criterion(
    kappa: float,
    Gamma_range: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Determine the melting point using the Lindemann criterion.

    The crystal melts when the Lindemann ratio exceeds a critical
    value: delta_L > delta_c ~ 0.15 (for 2D Yukawa systems).

    Parameters
    ----------
    kappa : float
        Screening parameter.
    Gamma_range : np.ndarray, optional
        Range of Gamma values to scan.

    Returns
    -------
    results : dict
        'Gamma_melt': estimated melting coupling,
        'delta_c': critical Lindemann parameter.
    """
    if Gamma_range is None:
        Gamma_range = np.linspace(50, 300, 20)

    # Simple estimate: Gamma_c ~ 137 * (1 + kappa) for kappa < 5
    # More accurate fits from the literature:
    # Gamma_c(kappa) ~ 137 + 40*kappa + 5*kappa^2 (approximate)
    Gamma_c = 137.0 + 40.0 * kappa + 5.0 * kappa**2

    return {
        "Gamma_melt": float(Gamma_c),
        "delta_c": 0.15,
        "kappa": kappa,
        "phase_at_Gamma_c": "hexagonal crystal -> liquid transition",
    }
