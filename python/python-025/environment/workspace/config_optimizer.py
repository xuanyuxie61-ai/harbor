"""
config_optimizer.py
===================
Configuration optimization and structural analysis synthesized from seed projects:
  - 1367_tsp_random (Monte Carlo random search heuristic)
  - 669_levenshtein_matrix (dynamic programming for edit distance)

Core algorithms:
  - Total Yukawa interaction energy computation
  - Monte Carlo energy minimization / relaxation
  - Metropolis acceptance criterion for finite temperature
  - Configuration distance via dynamic programming (adapted Levenshtein)
"""

import numpy as np
from yukawa_physics import yukawa_potential


def total_yukawa_energy(positions, Q_eff, lambda_D):
    """
    Compute total pairwise Yukawa interaction energy:
      E = sum_{i<j} U(|r_i - r_j|)
    
    where U(r) = (Q_eff^2 / 4*pi*eps0*r) * exp(-r/lambda_D).
    """
    N = positions.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-15:
                energy += yukawa_potential(r, Q_eff, lambda_D)
    return energy


def monte_carlo_relax(positions, Q_eff, lambda_D, n_steps, step_size, T=0.0, box_size=None):
    """
    Monte Carlo energy minimization based on seed 1367_tsp_random.
    
    Algorithm:
      For step = 1 .. n_steps:
        1. Randomly select a particle i
        2. Propose a random displacement within [-step_size, +step_size]^3
        3. Compute energy change delta_E
        4. If delta_E < 0: accept move
           Else if T > 0: accept with probability exp(-delta_E / k_B*T)
           Else: reject move
        5. Track best configuration found
    
    At T = 0, this is a greedy energy minimization.
    At T > 0, this is a simulated annealing / Metropolis Monte Carlo.
    """
    N = positions.shape[0]
    pos = positions.copy()
    current_energy = total_yukawa_energy(pos, Q_eff, lambda_D)
    k_B = 1.380649e-23
    
    best_energy = current_energy
    best_pos = pos.copy()
    accept_count = 0
    
    half_box = None
    if box_size is not None:
        half_box = box_size / 2.0
    
    for step in range(n_steps):
        i = np.random.randint(N)
        displacement = step_size * (2.0 * np.random.rand(3) - 1.0)
        old_pos = pos[i].copy()
        pos[i] += displacement
        
        # Optional boundary condition
        if half_box is not None:
            for d in range(3):
                if pos[i, d] > half_box:
                    pos[i, d] = half_box
                elif pos[i, d] < -half_box:
                    pos[i, d] = -half_box
        
        new_energy = total_yukawa_energy(pos, Q_eff, lambda_D)
        delta_E = new_energy - current_energy
        
        accepted = False
        if delta_E < 0.0:
            accepted = True
        elif T > 0.0:
            if np.random.rand() < np.exp(-delta_E / (k_B * T)):
                accepted = True
        
        if accepted:
            current_energy = new_energy
            accept_count += 1
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
        else:
            pos[i] = old_pos
    
    return best_pos, best_energy, accept_count


def configuration_distance(config1, config2):
    """
    Compute structural distance between two particle configurations using
    dynamic programming adapted from seed 669_levenshtein_matrix.
    
    Standard Levenshtein distance measures minimum edit operations to transform
    one string into another. Here, we adapt it to measure the minimum "cost"
    to align two 1D-ordered particle chains.
    
    Algorithm:
      1. Sort both configurations by x-coordinate to obtain 1D chains
      2. Build DP matrix d where d[i,j] = min cost to align first i particles
         of chain 1 with first j particles of chain 2
      3. Recurrence:
           d[i,j] = min( d[i-1,j]   + 1.0,       (deletion)
                         d[i,j-1]   + 1.0,       (insertion)
                         d[i-1,j-1] + ||c1[i]-c2[j]|| )  (substitution/match)
    
    Returns the minimum alignment cost.
    """
    # Sort by x-coordinate to establish 1D ordering
    c1 = config1[np.argsort(config1[:, 0])]
    c2 = config2[np.argsort(config2[:, 0])]
    
    m = min(len(c1), len(c2))
    if m == 0:
        return 0.0
    
    # Truncate to same length for clean comparison
    c1 = c1[:m]
    c2 = c2[:m]
    
    # Levenshtein-like DP
    d = np.zeros((m + 1, m + 1), dtype=float)
    for i in range(m + 1):
        d[i, 0] = float(i)
    for j in range(m + 1):
        d[0, j] = float(j)
    
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(c1[i-1] - c2[j-1])
            d[i, j] = min(
                d[i-1, j] + 1.0,
                d[i, j-1] + 1.0,
                d[i-1, j-1] + cost
            )
    
    return float(d[m, m])
