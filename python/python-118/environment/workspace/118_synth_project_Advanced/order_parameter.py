
import numpy as np
from scipy.special import sph_harm
from config import R_CUTOFF, STEINHARDT_L, Q6_THRESHOLD_SOLID, Q6_THRESHOLD_LIQUID






def compute_steinhardt_parameters(positions, box, neighbors, dists_sq, l=STEINHARDT_L):
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
            

            dr = positions[j] - positions[i]
            dr -= box * np.round(dr / box)
            

            x, y, z = dr
            

            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < 1e-10:
                continue
            theta = np.arccos(np.clip(z / r_norm, -1.0, 1.0))
            

            phi = np.arctan2(y, x)
            

            for m_idx, m in enumerate(range(-l, l + 1)):
                Q_lm[i, m_idx] += sph_harm(m, l, phi, theta)
        

        Q_lm[i] /= n_bonds
        

        norm_sq = np.sum(np.abs(Q_lm[i]) ** 2)
        Q_l[i] = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0) * norm_sq)
    
    return Q_l, Q_lm


def identify_phases(Q_l, threshold_solid=Q6_THRESHOLD_SOLID, threshold_liquid=Q6_THRESHOLD_LIQUID):
    is_solid = Q_l > threshold_solid
    is_liquid = Q_l < threshold_liquid
    is_interface = ~(is_solid | is_liquid)
    
    return is_solid, is_liquid, is_interface


def compute_density_profile(positions, box, species=None, n_bins=50):
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
    rho_mid = 0.5 * (rho_solid + rho_liquid)
    

    diff = rho_profile - rho_mid
    

    drhodz = np.gradient(rho_profile, z_centers)
    idx_max = np.argmax(np.abs(drhodz))
    
    z_interface = z_centers[idx_max]
    



    max_grad = np.abs(drhodz[idx_max])
    if max_grad > 1e-10:
        width = abs(rho_solid - rho_liquid) / (2.0 * max_grad)
    else:
        width = 5.0
    
    return z_interface, width


def compute_interface_width(positions, Q_l, box, n_bins=50):
    z = positions[:, 2]
    z_min, z_max = -box[2] / 2, box[2] / 2
    
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    

    Q_binned = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (z >= bin_edges[b]) & (z < bin_edges[b + 1])
        if np.any(mask):
            Q_binned[b] = np.mean(Q_l[mask])
    
    z_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    Q_solid = np.percentile(Q_l, 95)
    Q_liquid = np.percentile(Q_l, 5)
    

    Q_norm = (Q_binned - Q_liquid) / (Q_solid - Q_liquid + 1e-10)
    Q_norm = np.clip(Q_norm, 0.0, 1.0)
    

    try:
        idx_10 = np.where(Q_norm >= 0.1)[0][0]
        idx_90 = np.where(Q_norm >= 0.9)[0][0]
        width_10_90 = abs(z_centers[idx_10] - z_centers[idx_90])
    except IndexError:
        width_10_90 = box[2] * 0.2
    

    try:
        idx_50 = np.where(Q_norm >= 0.5)[0][0]
        z_interface = z_centers[idx_50]
    except IndexError:
        z_interface = 0.0
    
    return width_10_90, z_interface






def common_neighbor_analysis(positions, box, neighbors, dists_sq):
    n_atoms = positions.shape[0]
    structure_types = np.zeros(n_atoms, dtype=int)
    

    adjacency = [set(neighbors[i]) for i in range(n_atoms)]
    
    for i in range(n_atoms):
        fcc_count = 0
        hcp_count = 0
        bcc_count = 0
        
        for j in neighbors[i]:

            common = adjacency[i] & adjacency[j]
            n_common = len(common)
            

            m_bonds = 0
            common_list = list(common)
            for idx_a, a in enumerate(common_list):
                for b in common_list[idx_a + 1:]:
                    if b in adjacency[a]:
                        m_bonds += 1
            

            if n_common == 4 and m_bonds == 2:
                fcc_count += 1
            elif n_common == 4 and m_bonds == 4:
                bcc_count += 1
            elif n_common == 6 and m_bonds == 6:
                bcc_count += 1
        
        n_bonds = len(neighbors[i])
        if n_bonds == 0:
            structure_types[i] = 4
        elif fcc_count >= 6 and bcc_count < 3:
            structure_types[i] = 1
        elif bcc_count >= 4:
            structure_types[i] = 3
        else:
            structure_types[i] = 4
    
    return structure_types
