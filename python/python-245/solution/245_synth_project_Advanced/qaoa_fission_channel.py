"""
qaoa_fission_channel.py
=========================
Quantum-inspired combinatorial optimisation for determining optimal
fission channel configurations.

Maps from: 1196_QutacQuantum_Knapsack (QAOA for multi-knapsack via
           QUBO/Ising formulation)

Physical context
----------------
The fission channel problem: given a set of possible fragment pairs
(A_i, Z_i) that conserve (A_cn, Z_cn), find the combination that
minimises the total potential energy while satisfying conservation
laws and physical constraints.

This is formulated as a constrained optimisation problem analogous
to the multi-knapsack problem:

Minimise:  E_total = sum_i x_i * E_i + sum_{i<j} x_i * x_j * V_{ij}
Subject to:
  sum_i x_i * A_i = A_cn  (mass conservation)
  sum_i x_i * Z_i = Z_cn  (charge conservation)
  x_i in {0, 1}            (binary: channel open/closed)

Converted to QUBO (Quadratic Unconstrained Binary Optimisation):
  H = H_obj + lambda_A * H_mass + lambda_Z * H_charge

where H_obj encodes the energy, and the constraint terms penalise
violations of mass/charge conservation.

The QAOA algorithm applies alternating layers of cost and mixer
unitaries to find the ground state (optimal channel configuration).

For this classical simulation, we use a simulated-annealing-like
approach inspired by the QAOA structure.
"""

import math
import random
from typing import List, Dict, Tuple, Optional

from nuclear_constants import (
    q_value_fission, coulomb_barrier_energy, get_atomic_mass,
    ATOMIC_MASS_UNIT_MEV, RADIUS_PARAMETER
)


class FissionChannelOptimizer:
    """
    QAOA-inspired optimisation for fission channel selection.

    From 1196_QutacQuantum_Knapsack: QUBO formulation + classical
    optimisation (SLSQP-like) to simulate QAOA behaviour.
    """

    def __init__(self, z_cn: int = 92, a_cn: int = 236):
        self.Z = z_cn
        self.A = a_cn
        self.channels: List[Dict] = []
        self.n_channels = 0

    def generate_candidate_channels(self, n_light_range: Tuple[int, int] = (75, 115),
                                     n_channels: int = 20) -> None:
        """
        Generate candidate fission channels (fragment pairs).

        Each channel i has:
        - A_light_i, Z_light_i, A_heavy_i, Z_heavy_i
        - Q_value_i (energy release)
        - TKE_i (total kinetic energy)
        - probability weight
        """
        self.channels = []
        random.seed(42)

        for i in range(n_channels):
            a_light = random.randint(n_light_range[0], n_light_range[1])
            a_heavy = self.A - a_light

            # UCD for charges
            z_light = max(1, min(a_light - 1,
                                int(round(self.Z * a_light / self.A))))
            z_heavy = self.Z - z_light

            q_val = q_value_fission(self.A, self.Z,
                                    a_light, z_light, a_heavy, z_heavy, 0)
            tke = coulomb_barrier_energy(z_light, z_heavy, a_light, a_heavy)

            self.channels.append({
                'index': i,
                'a_light': a_light,
                'z_light': z_light,
                'a_heavy': a_heavy,
                'z_heavy': z_heavy,
                'q_value': q_val,
                'tke': tke,
                'excitation': q_val - tke,
            })

        self.n_channels = len(self.channels)

    def compute_qubo_coefficients(self, lambda_a: float = 10.0,
                                   lambda_z: float = 10.0) -> Tuple[List[float],
                                                                      List[List[float]]]:
        """
        Convert fission channel problem to QUBO form:
          H = sum_i h_i * x_i + sum_{i<j} J_{ij} * x_i * x_j

        Objective (H_obj): minimise -Q_value (maximise energy release)
        Constraint (H_mass): lambda_A * (sum_i A_light_i * x_i - A_target)^2
        Constraint (H_charge): lambda_Z * (sum_i Z_light_i * x_i - Z_target)^2

        From 1196_QutacQuantum_Knapsack: QUBO formulation via pyqubo concept.
        """
        n = self.n_channels
        h = [0.0] * n  # linear terms
        J = [[0.0] * n for _ in range(n)]  # quadratic terms

        for i in range(n):
            ch = self.channels[i]
            # Objective: prefer channels with large Q-value
            h[i] += -ch['q_value']

            # Mass conservation penalty: lambda_A * (A_i * x_i - A_cn/2)^2
            # Expanding: lambda_A * A_i^2 * x_i - lambda_A * A_cn * A_i * x_i + const
            a_target = self.A / 2.0
            h[i] += lambda_a * ch['a_light'] * (ch['a_light'] - 2.0 * a_target)

            # Charge conservation penalty
            z_target = self.Z / 2.0
            h[i] += lambda_z * ch['z_light'] * (ch['z_light'] - 2.0 * z_target)

        # Quadratic coupling (mass constraint cross terms)
        for i in range(n):
            for j in range(i + 1, n):
                ai = self.channels[i]['a_light']
                aj = self.channels[j]['a_light']
                zi = self.channels[i]['z_light']
                zj = self.channels[j]['z_light']

                J[i][j] = lambda_a * 2.0 * ai * aj + lambda_z * 2.0 * zi * zj

        return h, J

    def evaluate_hamiltonian(self, x: List[int],
                             h: List[float],
                             J: List[List[float]]) -> float:
        """
        Evaluate QUBO Hamiltonian for a given binary assignment x.
        H = sum_i h_i * x_i + sum_{i<j} J_{ij} * x_i * x_j
        """
        energy = 0.0
        for i in range(len(x)):
            if x[i] == 1:
                energy += h[i]
                for j in range(i + 1, len(x)):
                    if x[j] == 1:
                        energy += J[i][j]
        return energy

    def check_constraints(self, x: List[int]) -> Dict[str, any]:
        """
        Check mass and charge conservation for selected channels.
        """
        a_total = 0
        z_total = 0
        selected = []

        for i, xi in enumerate(x):
            if xi == 1:
                a_total += self.channels[i]['a_light']
                z_total += self.channels[i]['z_light']
                selected.append(i)

        a_error = abs(a_total - self.A / 2.0)
        z_error = abs(z_total - self.Z / 2.0)

        return {
            'a_total': a_total,
            'z_total': z_total,
            'a_error': a_error,
            'z_error': z_error,
            'satisfied': a_error < 1.0 and z_error < 1.0,
            'selected_channels': selected,
        }

    def simulated_annealing_solve(self, n_iterations: int = 500,
                                   t_initial: float = 5.0,
                                   t_final: float = 0.1,
                                   seed: int = 42) -> Dict[str, any]:
        """
        Simulated annealing solver (classical analogue of QAOA).

        From 1196: the QAOA uses SLSQP to optimise angles gamma, beta.
        Here we use SA to find the binary assignment minimising H.

        This simulates what QAOA would achieve with sufficient depth.
        """
        random.seed(seed)
        h, J_mat = self.compute_qubo_coefficients()
        n = self.n_channels

        # Initial random assignment
        x = [random.randint(0, 1) for _ in range(n)]
        energy = self.evaluate_hamiltonian(x, h, J_mat)

        best_x = x[:]
        best_energy = energy

        for iteration in range(n_iterations):
            # Temperature schedule (exponential cooling)
            frac = iteration / max(1, n_iterations - 1)
            temp = t_initial * (t_final / t_initial) ** frac

            # Propose flip
            flip_idx = random.randint(0, n - 1)
            x_new = x[:]
            x_new[flip_idx] = 1 - x_new[flip_idx]

            energy_new = self.evaluate_hamiltonian(x_new, h, J_mat)
            delta_e = energy_new - energy

            # Metropolis acceptance (from MH sampler concept)
            if delta_e < 0:
                accept = True
            else:
                if temp > 1e-10:
                    accept = random.random() < math.exp(-delta_e / temp)
                else:
                    accept = False

            if accept:
                x = x_new
                energy = energy_new

                if energy < best_energy:
                    best_x = x[:]
                    best_energy = energy

        # Check constraints on best solution
        constraints = self.check_constraints(best_x)

        return {
            'best_energy': best_energy,
            'best_assignment': best_x,
            'constraints': constraints,
            'n_iterations': n_iterations,
            'n_selected': sum(best_x),
        }

    def qaoa_energy_landscape(self, n_angles: int = 10) -> List[Dict[str, float]]:
        """
        Compute the QAOA energy landscape for p=1 layer.

        For p=1 QAOA, the expectation value <H>(gamma, beta) is computed
        for a grid of (gamma, beta) angles.

        gamma: cost function rotation angle
        beta: mixer rotation angle

        From 1196: QAOA with configurable number of reps (layers).
        """
        h, J_mat = self.compute_qubo_coefficients()
        results = []

        for ig in range(n_angles):
            gamma = ig * math.pi / (n_angles - 1) if n_angles > 1 else 0.0
            for ib in range(n_angles):
                beta = ib * math.pi / (n_angles - 1) if n_angles > 1 else 0.0

                # Simplified QAOA expectation value
                # For classical simulation: approximate as
                # <H> ~ sum_i h_i * sin(2*beta)*sin(2*gamma*h_i)
                #       + coupling terms
                exp_h = 0.0
                for i in range(self.n_channels):
                    # Single-qubit contribution
                    exp_h += h[i] * math.sin(2.0 * beta) * math.sin(2.0 * gamma * h[i]) * 0.5

                    # Two-qubit contributions (simplified)
                    for j in range(i + 1, self.n_channels):
                        exp_h += J_mat[i][j] * math.sin(2.0 * beta) ** 2 * 0.25

                results.append({
                    'gamma': gamma,
                    'beta': beta,
                    'expectation': exp_h,
                })

        return results

    def solve_and_analyse(self) -> Dict[str, any]:
        """
        Full solve and analysis pipeline.
        """
        self.generate_candidate_channels()

        # Solve via SA
        result = self.simulated_annealing_solve()

        # QAOA landscape
        landscape = self.qaoa_energy_landscape(8)

        # Find optimal angles
        if landscape:
            best_landscape = min(landscape, key=lambda x: x['expectation'])
        else:
            best_landscape = {'gamma': 0, 'beta': 0, 'expectation': 0}

        return {
            'optimization': result,
            'qaoa_best_angles': {
                'gamma': best_landscape['gamma'],
                'beta': best_landscape['beta'],
                'expectation': best_landscape['expectation'],
            },
            'n_candidate_channels': self.n_channels,
        }
