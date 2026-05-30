
import numpy as np






def hermite_cubic_coefficients(z1, z2, f1, d1, f2, d2):
    h = z2 - z1
    if abs(h) < 1e-14:
        raise ValueError("区间长度 h 必须为正")
    
    df = (f2 - f1) / h
    c0 = f1
    c1 = d1
    c2 = -(2.0 * d1 - 3.0 * df + d2) / h
    c3 = (d1 - 2.0 * df + d2) / (h * h)
    return c0, c1, c2, c3


def hermite_cubic_value(z1, z2, f1, d1, f2, d2, z_query):
    c0, c1, c2, c3 = hermite_cubic_coefficients(z1, z2, f1, d1, f2, d2)
    dz = z_query - z1
    
    f = c0 + dz * (c1 + dz * (c2 + dz * c3))
    df = c1 + dz * (2.0 * c2 + dz * 3.0 * c3)
    d2f = 2.0 * c2 + dz * 6.0 * c3
    d3f = 6.0 * c3
    return f, df, d2f, d3f


def hermite_cubic_integral(z1, z2, f1, d1, f2, d2):
    h = z2 - z1
    c0, c1, c2, c3 = hermite_cubic_coefficients(z1, z2, f1, d1, f2, d2)
    return h * (c0 + h / 2.0 * c1 + h**2 / 3.0 * c2 + h**3 / 4.0 * c3)






def build_hermite_spline(z_nodes, f_nodes, d_nodes):
    n = len(z_nodes)
    if len(f_nodes) != n or len(d_nodes) != n:
        raise ValueError("z_nodes, f_nodes, d_nodes 长度必须相等")
    if n < 2:
        raise ValueError("至少需要 2 个节点")
    

    if not np.all(np.diff(z_nodes) > 0):
        raise ValueError("z_nodes 必须严格单调递增")
    
    coeffs = []
    for k in range(n - 1):
        c0, c1, c2, c3 = hermite_cubic_coefficients(
            z_nodes[k], z_nodes[k+1],
            f_nodes[k], d_nodes[k],
            f_nodes[k+1], d_nodes[k+1]
        )
        coeffs.append((c0, c1, c2, c3))
    
    return {
        'z_nodes': z_nodes.copy(),
        'f_nodes': f_nodes.copy(),
        'd_nodes': d_nodes.copy(),
        'coeffs': coeffs,
        'n_segments': n - 1,
    }


def evaluate_hermite_spline(spline, z_query):
    z_nodes = spline['z_nodes']
    coeffs = spline['coeffs']
    
    is_scalar = np.isscalar(z_query)
    zq = np.atleast_1d(z_query)
    
    f = np.zeros_like(zq, dtype=float)
    df = np.zeros_like(zq, dtype=float)
    d2f = np.zeros_like(zq, dtype=float)
    d3f = np.zeros_like(zq, dtype=float)
    
    z_min = z_nodes[0]
    z_max = z_nodes[-1]
    
    for i, z in enumerate(zq):

        if z < z_min:
            seg = 0
        elif z >= z_max:
            seg = len(coeffs) - 1
        else:

            seg = np.searchsorted(z_nodes, z, side='right') - 1
            seg = max(0, min(seg, len(coeffs) - 1))
        
        c0, c1, c2, c3 = coeffs[seg]
        dz = z - z_nodes[seg]
        f[i] = c0 + dz * (c1 + dz * (c2 + dz * c3))
        df[i] = c1 + dz * (2.0 * c2 + dz * 3.0 * c3)
        d2f[i] = 2.0 * c2 + dz * 6.0 * c3
        d3f[i] = 6.0 * c3
    
    if is_scalar:
        return f[0], df[0], d2f[0], d3f[0]
    return f, df, d2f, d3f


def integrate_hermite_spline(spline, a, b):
    z_nodes = spline['z_nodes']
    coeffs = spline['coeffs']
    
    if a > b:
        return -integrate_hermite_spline(spline, b, a)
    
    total = 0.0
    n = len(z_nodes)
    
    for k in range(n - 1):
        z1, z2 = z_nodes[k], z_nodes[k+1]
        c0, c1, c2, c3 = coeffs[k]
        

        left = max(a, z1)
        right = min(b, z2)
        if left >= right:
            continue
        

        dl = left - z1
        dr = right - z1
        
        def F(dz):
            return c0 * dz + c1 * dz**2 / 2.0 + c2 * dz**3 / 3.0 + c3 * dz**4 / 4.0
        
        total += F(dr) - F(dl)
    
    return total






def estimate_derivatives_central(z_nodes, f_nodes):
    n = len(z_nodes)
    d_nodes = np.zeros(n)
    
    d_nodes[0] = (f_nodes[1] - f_nodes[0]) / (z_nodes[1] - z_nodes[0])
    d_nodes[-1] = (f_nodes[-1] - f_nodes[-2]) / (z_nodes[-1] - z_nodes[-2])
    
    for k in range(1, n - 1):
        d_nodes[k] = (f_nodes[k+1] - f_nodes[k-1]) / (z_nodes[k+1] - z_nodes[k-1])
    
    return d_nodes


def compute_brunt_vaisala_frequency(z_nodes, T_nodes, S_nodes, lat=30.0):
    n = len(z_nodes)
    if n < 2:
        raise ValueError("至少需要两个深度层")
    

    phi = np.radians(lat)
    g = 9.780327 * (1.0 + 0.0053024 * np.sin(phi)**2 - 0.0000058 * np.sin(2*phi)**2)
    
    rho0 = 1025.0
    alpha_T = -0.15
    beta_S = 0.78
    

    dT = estimate_derivatives_central(z_nodes, T_nodes)
    dS = estimate_derivatives_central(z_nodes, S_nodes)
    
    T_spline = build_hermite_spline(z_nodes, T_nodes, dT)
    S_spline = build_hermite_spline(z_nodes, S_nodes, dS)
    

    z_mid = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    _, dT_dz, _, _ = evaluate_hermite_spline(T_spline, z_mid)
    _, dS_dz, _, _ = evaluate_hermite_spline(S_spline, z_mid)
    
    drho_dz = alpha_T * dT_dz + beta_S * dS_dz
    

    N2 = -(g / rho0) * drho_dz
    

    N2 = np.where(N2 < -1e-6, N2, np.maximum(N2, 0.0))
    
    return N2, z_mid


def mixed_layer_depth(z_nodes, T_nodes, threshold=0.5):
    if len(z_nodes) < 2:
        return z_nodes[0] if len(z_nodes) > 0 else 0.0
    
    T_surface = T_nodes[0]
    T_target = T_surface - threshold
    

    for k in range(len(z_nodes) - 1):
        if (T_nodes[k] - T_target) * (T_nodes[k+1] - T_target) <= 0:

            z1, z2 = z_nodes[k], z_nodes[k+1]
            T1, T2 = T_nodes[k], T_nodes[k+1]
            if abs(T2 - T1) > 1e-10:
                frac = (T_target - T1) / (T2 - T1)
                return z1 + frac * (z2 - z1)
    
    return z_nodes[-1]
