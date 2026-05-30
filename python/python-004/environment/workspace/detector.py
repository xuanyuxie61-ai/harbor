
import numpy as np
from numpy.linalg import lstsq, norm






def detector_tensor(arm1, arm2):
    u = np.asarray(arm1, dtype=np.float64)
    v = np.asarray(arm2, dtype=np.float64)
    
    if u.shape != (3,) or v.shape != (3,):
        raise ValueError("臂方向必须为三维向量")
    

    u_norm = norm(u)
    v_norm = norm(v)
    if u_norm < 1e-12 or v_norm < 1e-12:
        raise ValueError("臂方向向量不能为零")
    u = u / u_norm
    v = v / v_norm
    

    dot_uv = np.dot(u, v)
    if np.abs(dot_uv) > 0.01:

        v = v - dot_uv * u
        v_norm = norm(v)
        if v_norm < 1e-12:
            raise ValueError("臂方向线性相关")
        v = v / v_norm
    
    D = 0.5 * (np.outer(u, u) - np.outer(v, v))
    return D


def antenna_pattern_functions(theta, phi, psi, arm1, arm2):
    theta = float(theta)
    phi = float(phi)
    psi = float(psi)
    

    n = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=np.float64)
    

    l = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ], dtype=np.float64)
    m = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=np.float64)
    

    lp = l * np.cos(psi) + m * np.sin(psi)
    mp = -l * np.sin(psi) + m * np.cos(psi)
    

    e_plus = np.outer(lp, lp) - np.outer(mp, mp)
    e_cross = np.outer(lp, mp) + np.outer(mp, lp)
    
    D = detector_tensor(arm1, arm2)
    
    F_plus = np.sum(D * e_plus)
    F_cross = np.sum(D * e_cross)
    
    return F_plus, F_cross






def spherical_distance(lat1, lon1, lat2, lon2, radius=1.0):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    a = min(1.0, max(0.0, a))
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def latlon_to_xyz(lat, lon, radius=1.0):
    x = radius * np.cos(lon) * np.cos(lat)
    y = radius * np.sin(lon) * np.cos(lat)
    z = radius * np.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def xyz_to_latlon(xyz, radius=1.0):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] != 3:
        raise ValueError("xyz 必须为三维向量")
    
    r = norm(xyz)
    if r < 1e-12:
        return 0.0, 0.0
    
    lat = np.arcsin(np.clip(xyz[2] / r, -1.0, 1.0))
    lon = np.arctan2(xyz[1], xyz[0])
    return lat, lon


def compute_sky_position_from_time_delays(detector_positions, time_delays, radius=1.0, max_iter=100):
    detector_positions = np.asarray(detector_positions, dtype=np.float64)
    time_delays = np.asarray(time_delays, dtype=np.float64)
    
    N = detector_positions.shape[0]
    if N < 3:
        raise ValueError("至少需要 3 个探测器进行天球定位")
    if time_delays.shape[0] != N:
        raise ValueError("time_delays 长度必须与探测器数量一致")
    
    c = 1.0
    


    n_constraints = N - 1
    A = np.zeros((n_constraints, 3), dtype=np.float64)
    b = np.zeros(n_constraints, dtype=np.float64)
    
    for i in range(1, N):
        A[i - 1, :] = (detector_positions[i, :] - detector_positions[0, :]) / c
        b[i - 1] = time_delays[i] - time_delays[0]
    

    n_est, residuals, rank, s = lstsq(A, b, rcond=None)
    

    n_norm = norm(n_est)
    if n_norm < 1e-12:
        n_est = np.array([1.0, 0.0, 0.0])
    else:
        n_est = n_est / n_norm
    

    for _ in range(max_iter):

        residual = A.dot(n_est) - b
        

        J = np.vstack([A, 2.0 * n_est.reshape(1, 3)])
        rhs = np.hstack([-residual, [1.0 - norm(n_est)**2]])
        
        delta, _, _, _ = lstsq(J, rhs, rcond=None)
        n_est = n_est + delta
        

        n_norm = norm(n_est)
        if n_norm > 1e-12:
            n_est = n_est / n_norm
        
        if norm(delta) < 1e-12:
            break
    
    lat, lon = xyz_to_latlon(n_est, radius)
    return lat, lon, n_est


def network_snr(F_plus_list, F_cross_list, h_plus, h_cross, noise_psd=1.0):
    rho_sq = 0.0
    for Fp, Fc in zip(F_plus_list, F_cross_list):
        h_detector = Fp * h_plus + Fc * h_cross
        rho_sq += np.sum(np.abs(h_detector)**2) / noise_psd
    
    return np.sqrt(max(rho_sq, 0.0))






LIGO_HANFORD_ARMS = {
    'name': 'LIGO Hanford',
    'arm1': np.array([-0.2239, 0.7998, 0.5569], dtype=np.float64),
    'arm2': np.array([-0.9140, 0.0261, -0.4049], dtype=np.float64),
    'position': np.array([-2.1614, -3.8347, 4.6005], dtype=np.float64) * 1e6 / 299792458.0
}

LIGO_LIVINGSTON_ARMS = {
    'name': 'LIGO Livingston',
    'arm1': np.array([-0.9546, -0.1416, 0.2622], dtype=np.float64),
    'arm2': np.array([0.2977, -0.4879, 0.8205], dtype=np.float64),
    'position': np.array([-7.4276, -0.2470, 0.5849], dtype=np.float64) * 1e6 / 299792458.0
}

VIRGO_ARMS = {
    'name': 'Virgo',
    'arm1': np.array([-0.7005, 0.2085, 0.6826], dtype=np.float64),
    'arm2': np.array([-0.0538, -0.9691, 0.2408], dtype=np.float64),
    'position': np.array([4.5464, 0.8429, 0.9877], dtype=np.float64) * 1e6 / 299792458.0
}


def get_standard_detector_network():
    return [LIGO_HANFORD_ARMS, LIGO_LIVINGSTON_ARMS, VIRGO_ARMS]
