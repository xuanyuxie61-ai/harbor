
import numpy as np


def lindemann_parameter(positions, equilibrium_positions, a_ws):
    displacements = positions - equilibrium_positions
    u2 = np.mean(np.sum(displacements**2, axis=1))
    return np.sqrt(max(u2, 0.0)) / a_ws


def radial_distribution_function(positions, dr, r_max):
    N = positions.shape[0]
    dim = positions.shape[1]
    n_bins = max(1, int(r_max / dr))
    g = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < r_max:
                idx = int(r / dr)
                if idx < n_bins:
                    g[idx] += 2.0
    

    r_bins = np.arange(n_bins) * dr + dr / 2.0
    for i in range(n_bins):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        if dim == 2:
            shell_vol = np.pi * (r_outer**2 - r_inner**2)
        else:
            shell_vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        if shell_vol > 0:
            g[i] /= (N * shell_vol)
    
    return r_bins, g


def structure_factor(q, positions):
    N = positions.shape[0]
    total = 0.0 + 0.0j
    for j in range(N):
        qr = np.dot(q, positions[j])
        total += np.cos(qr) + 1j * np.sin(qr)
    return float(np.abs(total)**2 / N)


def detect_phase(gamma, lindemann, gamma_c=170.0, lindemann_c=0.10):
    if gamma > gamma_c and lindemann < lindemann_c:
        return "CRYSTALLINE"
    elif gamma < 0.5 * gamma_c or lindemann > 2.0 * lindemann_c:
        return "GASEOUS"
    else:
        return "LIQUID"


def compute_bond_orientational_order(positions, n_neighbors=6):
    N = positions.shape[0]
    if N == 0:
        return 0.0
    
    psi_sum = 0.0
    for j in range(N):

        dists = np.array([np.linalg.norm(positions[j] - positions[k]) 
                          for k in range(N) if k != j])
        if len(dists) == 0:
            continue
        cutoff = np.sort(dists)[min(n_neighbors, len(dists)) - 1] * 1.1
        
        local_psi = 0.0 + 0.0j
        n_count = 0
        for k in range(N):
            if j == k:
                continue
            r_jk = positions[k, :2] - positions[j, :2]
            r = np.linalg.norm(r_jk)
            if r < 1e-10:
                continue
            if r <= cutoff:
                theta = np.arctan2(r_jk[1], r_jk[0])
                local_psi += np.cos(6.0 * theta) + 1j * np.sin(6.0 * theta)
                n_count += 1
        
        if n_count > 0:
            psi_sum += np.abs(local_psi / n_count)
    
    return psi_sum / N


def debye_waller_factor(q_vec, u_rms):
    q2 = np.dot(q_vec, q_vec)
    return np.exp(-0.5 * q2 * u_rms**2)
