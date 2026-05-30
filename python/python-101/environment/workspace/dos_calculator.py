
import numpy as np
from physics_core import C_0, local_density_of_states_3d






def simplex_unit_sample(m, n_samples):
    if m < 1 or n_samples < 1:
        raise ValueError("维数和采样数必须 >= 1")
    
    x = np.zeros((m, n_samples))
    for j in range(n_samples):
        e = -np.log(np.random.rand(m + 1))
        total = np.sum(e)
        if total < 1e-18:
            total = 1.0
        x[:, j] = e[:m] / total
    
    return x


def simplex_unit_to_general(m, n_samples, t_vertices, ref_points):
    phy = np.zeros((m, n_samples))
    for dim in range(m):
        phy[dim, :] = t_vertices[dim, 0]
        for vertex in range(1, m + 1):
            phy[dim, :] += (t_vertices[dim, vertex] - t_vertices[dim, 0]) * ref_points[vertex - 1, :]
    
    return phy


def simplex_general_sample(m, n_samples, t_vertices):
    ref = simplex_unit_sample(m, n_samples)
    return simplex_unit_to_general(m, n_samples, t_vertices, ref)


def simplex_volume(m, t_vertices):
    if m < 1:
        raise ValueError("维数必须 >= 1")
    
    T = np.zeros((m, m))
    for i in range(m):
        T[:, i] = t_vertices[:, i + 1] - t_vertices[:, 0]
    
    det = np.linalg.det(T)
    factorial = 1
    for i in range(2, m + 1):
        factorial *= i
    
    return abs(det) / factorial


def monte_carlo_dos_brillouin(omega_bands, k_points, n_samples=10000):
    N_k, n_bands = omega_bands.shape
    if N_k < 2:
        raise ValueError("k 点数量必须 >= 2")
    

    omega_min = np.min(omega_bands)
    omega_max = np.max(omega_bands)
    if omega_max <= omega_min:
        return np.array([omega_min]), np.array([0.0])
    
    n_bins = max(50, N_k // 2)
    omega_bins = np.linspace(omega_min, omega_max, n_bins)
    dos = np.zeros(n_bins)
    

    domega = (omega_max - omega_min) / n_bins
    sigma = max(domega * 2.0, 1e-12)
    

    k_min = np.min(k_points, axis=0)
    k_max = np.max(k_points, axis=0)
    area_bz = (k_max[0] - k_min[0]) * (k_max[1] - k_min[1])
    
    if area_bz < 1e-18:
        return omega_bins, dos
    

    k_samples = np.random.uniform(k_min, k_max, size=(n_samples, 2))
    

    for ks in k_samples:

        distances = np.sum((k_points - ks) ** 2, axis=1)
        idx_sorted = np.argsort(distances)[:4]
        
        if np.sum(distances[idx_sorted]) < 1e-18:
            weights = np.ones(len(idx_sorted)) / len(idx_sorted)
        else:
            inv_dist = 1.0 / (distances[idx_sorted] + 1e-12)
            weights = inv_dist / np.sum(inv_dist)
        
        for band in range(n_bands):
            omega_interp = np.sum(omega_bands[idx_sorted, band] * weights)

            dos += np.exp(-((omega_bins - omega_interp) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    
    dos /= (n_samples * n_bands)
    
    return omega_bins, dos






def set_discrete_cdf(n1, n2, pdf):
    if n1 < 1 or n2 < 1:
        raise ValueError("维度必须 >= 1")
    pdf = np.asarray(pdf)
    if pdf.shape != (n1, n2):
        raise ValueError("pdf 形状不匹配")
    if np.any(pdf < 0):
        raise ValueError("PDF 值必须非负")
    
    cdf = np.zeros((n1, n2))
    total = 0.0
    for j in range(n2):
        for i in range(n1):
            total += pdf[i, j]
            cdf[i, j] = total
    

    if total > 1e-18:
        cdf /= total
    
    return cdf


def discrete_cdf_to_xy(n1, n2, cdf, n_samples, u=None):
    if u is None:
        u = np.random.rand(n_samples)
    else:
        u = np.asarray(u)
        if len(u) != n_samples:
            raise ValueError("u 长度必须等于 n_samples")
    
    u = np.clip(u, 0.0, 1.0)
    xy = np.zeros((2, n_samples))
    
    cdf_flat = cdf.flatten()
    
    for k in range(n_samples):

        idx = np.searchsorted(cdf_flat, u[k])
        idx = min(idx, n1 * n2 - 1)
        i = idx % n1
        j = idx // n1
        

        r = np.random.rand(2)
        xy[0, k] = (i + r[0]) / n1
        xy[1, k] = (j + r[1]) / n2
    
    return xy


def importance_sampled_dos(omega_bands, k_points, n1=20, n2=20, n_samples=5000):
    N_k, n_bands = omega_bands.shape
    

    kx_min, kx_max = np.min(k_points[:, 0]), np.max(k_points[:, 0])
    ky_min, ky_max = np.min(k_points[:, 1]), np.max(k_points[:, 1])
    
    pdf = np.zeros((n1, n2))
    for band in range(n_bands):
        for ik in range(N_k):
            i = int(np.clip((k_points[ik, 0] - kx_min) / (kx_max - kx_min) * (n1 - 1), 0, n1 - 1))
            j = int(np.clip((k_points[ik, 1] - ky_min) / (ky_max - ky_min) * (n2 - 1), 0, n2 - 1))
            pdf[i, j] += np.sum(omega_bands[ik, band])
    
    pdf = np.maximum(pdf, 1e-12)
    cdf = set_discrete_cdf(n1, n2, pdf)
    

    u = np.random.rand(n_samples)
    xy_samples = discrete_cdf_to_xy(n1, n2, cdf, n_samples, u)
    

    k_samples = np.zeros((n_samples, 2))
    k_samples[:, 0] = kx_min + xy_samples[0, :] * (kx_max - kx_min)
    k_samples[:, 1] = ky_min + xy_samples[1, :] * (ky_max - ky_min)
    

    omega_min = np.min(omega_bands)
    omega_max = np.max(omega_bands)
    n_bins = 50
    omega_bins = np.linspace(omega_min, omega_max, n_bins)
    dos = np.zeros(n_bins)
    
    sigma = (omega_max - omega_min) / n_bins * 2.0
    
    for ks in k_samples:
        distances = np.sum((k_points - ks) ** 2, axis=1)
        idx_sorted = np.argsort(distances)[:4]
        
        inv_dist = 1.0 / (distances[idx_sorted] + 1e-12)
        weights = inv_dist / np.sum(inv_dist)
        
        for band in range(n_bands):
            omega_interp = np.sum(omega_bands[idx_sorted, band] * weights)
            dos += np.exp(-((omega_bins - omega_interp) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    
    dos /= (n_samples * n_bands)
    
    return omega_bins, dos






def van_hove_singularity_type(omega_bands, k_path_distance):
    N_k, n_bands = omega_bands.shape
    if N_k < 5:
        return []
    
    singularities = []
    
    for band in range(n_bands):
        omega = omega_bands[:, band]
        k = k_path_distance
        

        d2omega = np.zeros(N_k)
        dk = k[1] - k[0]
        if abs(dk) < 1e-18:
            continue
        
        d2omega[0] = (omega[2] - 2 * omega[1] + omega[0]) / dk ** 2
        d2omega[-1] = (omega[-1] - 2 * omega[-2] + omega[-3]) / dk ** 2
        for i in range(1, N_k - 1):
            d2omega[i] = (omega[i + 1] - 2 * omega[i] + omega[i - 1]) / dk ** 2
        

        for i in range(N_k - 1):
            if d2omega[i] * d2omega[i + 1] < 0:

                t = abs(d2omega[i]) / (abs(d2omega[i]) + abs(d2omega[i + 1]))
                k_sing = k[i] + t * (k[i + 1] - k[i])
                omega_sing = omega[i] + t * (omega[i + 1] - omega[i])
                

                if d2omega[i] > 0:
                    vh_type = 'M1_saddle'
                else:
                    vh_type = 'M0_or_M2_extremum'
                
                singularities.append({
                    'band': band,
                    'k_position': k_sing,
                    'omega': omega_sing,
                    'type': vh_type,
                    'curvature_change': 'positive_to_negative' if d2omega[i] > 0 else 'negative_to_positive'
                })
    
    return singularities
