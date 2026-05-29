"""
order_parameter.py

Order parameter analysis for solid-liquid interface identification.

Synthesizes concepts from:
    - 081_besselzero: Bessel function zeros for radial basis
    - 463_gegenbauer_rule: Angular integration on sphere
    - 641_laguerre_polynomial: Radial basis functions
    
Physical Model:
    Steinhardt Bond-Orientational Order Parameters:
    
    For each atom i, consider its N_b neighbors within a cutoff distance.
    The local order parameter is:
    
        Q_lm(i) = (1 / N_b) * sum_{j=1}^{N_b} Y_lm( theta_{ij}, phi_{ij} )
    
    where Y_lm are spherical harmonics and (theta, phi) are the spherical
    angles of the bond vector r_j - r_i relative to a global coordinate system.
    
    The rotationally invariant order parameter is:
    
        Q_l(i) = sqrt( 4*pi / (2*l+1) * sum_{m=-l}^{l} |Q_lm(i)|^2 )
    
    For a perfect FCC crystal:
        Q_4 = 0.19094
        Q_6 = 0.57452
    
    For a liquid:
        Q_4 ~ 0.0
        Q_6 ~ 0.0
    
    The interface can be located by finding the region where Q_6 transitions
    from the solid value to the liquid value.
    
    Additional Analysis:
        - Common Neighbor Analysis (CNA) for local structure identification
        - Density profile rho(z) for interface width
"""

import numpy as np
from scipy.special import sph_harm
from config import R_CUTOFF, STEINHARDT_L, Q6_THRESHOLD_SOLID, Q6_THRESHOLD_LIQUID


# =============================================================================
# Spherical Harmonics and Steinhardt Parameters
# =============================================================================

def compute_steinhardt_parameters(positions, box, neighbors, dists_sq, l=STEINHARDT_L):
    """
    Compute Steinhardt bond-orientational order parameters Q_l for each atom.
    
    Algorithm:
        For each atom i:
            1. Find neighbors j within cutoff
            2. Compute bond angles (theta_{ij}, phi_{ij})
            3. Compute Q_lm(i) = (1/N_b) * sum_j Y_lm(theta_{ij}, phi_{ij})
            4. Compute Q_l(i) = sqrt(4*pi/(2*l+1) * sum_m |Q_lm|^2)
    
    Args:
        positions: (N, 3) array
        box: (3,) array
        neighbors: list of neighbor lists
        dists_sq: list of squared distances
        l: spherical harmonic degree (typically 4 or 6)
        
    Returns:
        Q_l: (N,) array with order parameters
        Q_lm: (N, 2*l+1) complex array with spherical harmonic coefficients
    """
    n_atoms = positions.shape[0]
    n_m = 2 * l + 1
    
    Q_lm = np.zeros((n_atoms, n_m), dtype=complex)
    Q_l = np.zeros(n_atoms)
    
    for i in range(n_atoms):
        n_bonds = len(neighbors[i])
        if n_bonds == 0:
            Q_l[i] = 0.0
            continue
        
        for idx_j, r_sq in enumerate(dists_sq[i]):
            r = np.sqrt(r_sq)
            if r < 1e-6:
                continue
            
            j = neighbors[i][idx_j]
            
            # Bond vector with minimum image convention
            dr = positions[j] - positions[i]
            dr -= box * np.round(dr / box)
            
            # Spherical coordinates
            x, y, z = dr
            
            # Polar angle theta (from z-axis)
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < 1e-10:
                continue
            theta = np.arccos(np.clip(z / r_norm, -1.0, 1.0))
            
            # Azimuthal angle phi (in xy-plane)
            phi = np.arctan2(y, x)
            
            # Spherical harmonics for all m
            for m_idx, m in enumerate(range(-l, l + 1)):
                Q_lm[i, m_idx] += sph_harm(m, l, phi, theta)
        
        # Normalize by number of bonds
        Q_lm[i] /= n_bonds
        
        # Compute rotationally invariant Q_l
        norm_sq = np.sum(np.abs(Q_lm[i]) ** 2)
        Q_l[i] = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0) * norm_sq)
    
    return Q_l, Q_lm


def identify_phases(Q_l, threshold_solid=Q6_THRESHOLD_SOLID, threshold_liquid=Q6_THRESHOLD_LIQUID):
    """
    Identify solid and liquid atoms based on order parameter thresholds.
    
    Classification:
        Q_l > threshold_solid  => solid
        Q_l < threshold_liquid => liquid
        otherwise              => interface
    
    Args:
        Q_l: (N,) array of order parameters
        threshold_solid: solid threshold
        threshold_liquid: liquid threshold
        
    Returns:
        is_solid: (N,) boolean array
        is_liquid: (N,) boolean array
        is_interface: (N,) boolean array
    """
    is_solid = Q_l > threshold_solid
    is_liquid = Q_l < threshold_liquid
    is_interface = ~(is_solid | is_liquid)
    
    return is_solid, is_liquid, is_interface


def compute_density_profile(positions, box, species=None, n_bins=50):
    """
    Compute 1D density profile along z-axis.
    
    rho(z) = (1/A) * sum_i delta(z - z_i)
    
    where A = Lx * Ly is the cross-sectional area.
    
    Args:
        positions: (N, 3) array
        box: (3,) array
        species: (N,) array, if provided compute partial profiles
        n_bins: number of bins
        
    Returns:
        z_centers: bin centers
        rho_total: total density profile
        rho_A, rho_B: partial density profiles (if species provided)
    """
    z = positions[:, 2]
    z_min, z_max = -box[2] / 2, box[2] / 2
    
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    z_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    area = box[0] * box[1]
    
    counts, _ = np.histogram(z, bins=bin_edges)
    rho_total = counts / (area * bin_width)
    
    if species is not None:
        mask_A = species == 0
        mask_B = species == 1
        counts_A, _ = np.histogram(z[mask_A], bins=bin_edges)
        counts_B, _ = np.histogram(z[mask_B], bins=bin_edges)
        rho_A = counts_A / (area * bin_width)
        rho_B = counts_B / (area * bin_width)
        return z_centers, rho_total, rho_A, rho_B
    
    return z_centers, rho_total, None, None


def fit_interface_position(z_centers, rho_profile, rho_solid, rho_liquid):
    """
    Fit the interface position from the density profile.
    
    The density profile across a diffuse interface follows:
        rho(z) = (rho_solid + rho_liquid)/2 
               + (rho_solid - rho_liquid)/2 * tanh( (z - z0) / w )
    
    where z0 is the interface position and w is the interface width.
    
    We estimate z0 as the point where rho(z0) = (rho_solid + rho_liquid) / 2.
    
    Args:
        z_centers: bin centers
        rho_profile: density profile
        rho_solid: solid-phase density
        rho_liquid: liquid-phase density
        
    Returns:
        z_interface: estimated interface position
        width: estimated interface width
    """
    rho_mid = 0.5 * (rho_solid + rho_liquid)
    
    # Find crossing points
    diff = rho_profile - rho_mid
    
    # Find the steepest gradient region
    drhodz = np.gradient(rho_profile, z_centers)
    idx_max = np.argmax(np.abs(drhodz))
    
    z_interface = z_centers[idx_max]
    
    # Estimate width from gradient
    # drho/dz = (rho_s - rho_l)/(2*w) * sech^2((z-z0)/w)
    # At z = z0: drho/dz = (rho_s - rho_l)/(2*w)
    max_grad = np.abs(drhodz[idx_max])
    if max_grad > 1e-10:
        width = abs(rho_solid - rho_liquid) / (2.0 * max_grad)
    else:
        width = 5.0
    
    return z_interface, width


def compute_interface_width(positions, Q_l, box, n_bins=50):
    """
    Compute interface width from order parameter profile.
    
    The order parameter profile is:
        Q(z) = Q_solid + (Q_liquid - Q_solid) * phi(z)
    
    where phi(z) is the interface profile function.
    
    The interface width is often defined as the distance over which
    phi(z) changes from 0.1 to 0.9 (the 10-90 width).
    
    Args:
        positions: (N, 3) array
        Q_l: (N,) array of order parameters
        box: (3,) array
        n_bins: number of bins
        
    Returns:
        width_10_90: 10-90 interface width
        z_interface: interface position
    """
    z = positions[:, 2]
    z_min, z_max = -box[2] / 2, box[2] / 2
    
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    
    # Compute average Q in each bin
    Q_binned = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (z >= bin_edges[b]) & (z < bin_edges[b + 1])
        if np.any(mask):
            Q_binned[b] = np.mean(Q_l[mask])
    
    z_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    Q_solid = np.percentile(Q_l, 95)
    Q_liquid = np.percentile(Q_l, 5)
    
    # Normalize
    Q_norm = (Q_binned - Q_liquid) / (Q_solid - Q_liquid + 1e-10)
    Q_norm = np.clip(Q_norm, 0.0, 1.0)
    
    # Find 10% and 90% crossing
    try:
        idx_10 = np.where(Q_norm >= 0.1)[0][0]
        idx_90 = np.where(Q_norm >= 0.9)[0][0]
        width_10_90 = abs(z_centers[idx_10] - z_centers[idx_90])
    except IndexError:
        width_10_90 = box[2] * 0.2
    
    # Interface at 50%
    try:
        idx_50 = np.where(Q_norm >= 0.5)[0][0]
        z_interface = z_centers[idx_50]
    except IndexError:
        z_interface = 0.0
    
    return width_10_90, z_interface


# =============================================================================
# Common Neighbor Analysis (CNA)
# =============================================================================

def common_neighbor_analysis(positions, box, neighbors, dists_sq):
    """
    Perform Common Neighbor Analysis to identify local crystal structures.
    
    For each pair of bonded atoms (i, j), identify their common neighbors
    (atoms k that are bonded to both i and j).
    
    The CNA signature is (n, m, l) where:
        n = number of common neighbors
        m = number of bonds among common neighbors
        l = longest chain of bonds among common neighbors
    
    Typical signatures:
        FCC: (4, 2, 1) and (4, 2, 2)
        HCP: (4, 2, 1) and (4, 2, 2) and (2, 1, 1)
        BCC: (4, 4, 4) and (6, 6, 6)
        Liquid: disordered
    
    Args:
        positions: (N, 3) array
        box: (3,) array
        neighbors: list of neighbor lists
        dists_sq: list of squared distances
        
    Returns:
        structure_types: (N,) array with 0=unknown, 1=FCC, 2=HCP, 3=BCC, 4=liquid
    """
    n_atoms = positions.shape[0]
    structure_types = np.zeros(n_atoms, dtype=int)
    
    # Build adjacency sets for fast lookup
    adjacency = [set(neighbors[i]) for i in range(n_atoms)]
    
    for i in range(n_atoms):
        fcc_count = 0
        hcp_count = 0
        bcc_count = 0
        
        for j in neighbors[i]:
            # Common neighbors of i and j
            common = adjacency[i] & adjacency[j]
            n_common = len(common)
            
            # Count bonds among common neighbors
            m_bonds = 0
            common_list = list(common)
            for idx_a, a in enumerate(common_list):
                for b in common_list[idx_a + 1:]:
                    if b in adjacency[a]:
                        m_bonds += 1
            
            # Simple classification
            if n_common == 4 and m_bonds == 2:
                fcc_count += 1
            elif n_common == 4 and m_bonds == 4:
                bcc_count += 1
            elif n_common == 6 and m_bonds == 6:
                bcc_count += 1
        
        n_bonds = len(neighbors[i])
        if n_bonds == 0:
            structure_types[i] = 4  # liquid/unknown
        elif fcc_count >= 6 and bcc_count < 3:
            structure_types[i] = 1  # FCC
        elif bcc_count >= 4:
            structure_types[i] = 3  # BCC
        else:
            structure_types[i] = 4  # liquid/disordered
    
    return structure_types
