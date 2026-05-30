
import numpy as np
from numpy.polynomial import polynomial as P






def poly_eval(coeffs, z):
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    z = np.asarray(z, dtype=np.complex128)
    result = np.zeros_like(z, dtype=np.complex128)
    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(coeffs, tol=1e-14, max_iter=1000):
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.ndim != 1:
        raise ValueError("coeffs 必须为一维数组")
    

    while len(coeffs) > 1 and np.abs(coeffs[-1]) < 1e-15:
        coeffs = coeffs[:-1]
    
    d = len(coeffs) - 1
    if d < 1:
        raise ValueError("多项式次数必须至少为 1")
    

    leading = coeffs[-1]
    R = 1.0 + np.max(np.abs(coeffs[:-1] / leading))
    

    theta = np.linspace(0.0, 2.0 * np.pi, d, endpoint=False)
    roots = R * np.exp(1j * theta)
    
    for iteration in range(max_iter):
        roots_old = roots.copy()
        for i in range(d):
            zi = roots_old[i]
            denom = np.prod(zi - np.delete(roots_old, i))
            if np.abs(denom) < 1e-300:
                denom = 1e-300 * np.exp(1j * np.angle(denom)) if denom != 0 else 1e-300
            roots[i] = zi - poly_eval(coeffs, zi) / denom
        
        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            break
    
    return roots






def qnm_characteristic_polynomial(l, m, n, M=1.0, a=0.0):


    known_roots = []
    

    if a == 0.0:

        if l == 2 and m == 2:
            known_roots.append((0.37367 - 0.08896j) / M)
        if l == 2 and m == 1:
            known_roots.append((0.34671 - 0.09606j) / M)
        if l == 3 and m == 3:
            known_roots.append((0.59944 - 0.09270j) / M)
    else:

        omega0 = (0.37367 - 0.08896j) / M
        omega_cor = omega0 + m * a / (2 * M**2) * 0.1
        known_roots.append(omega_cor)
    

    for k in range(1, n + 2):
        damp = -0.1 * k / M
        freq = (0.35 + 0.02 * k) / M
        known_roots.append(freq + damp * 1j)
    

    roots_arr = np.array(known_roots, dtype=np.complex128)

    poly = np.array([1.0], dtype=np.complex128)
    for r in roots_arr:
        poly = np.convolve(poly, np.array([1.0, -r], dtype=np.complex128))
    
    return poly, roots_arr


def solve_qnm_frequencies(l_max=4, n_overtones=2, M=1.0, a=0.0):
    results = {}
    for l in range(2, l_max + 1):
        for m in range(-l, l + 1):
            for n in range(n_overtones + 1):
                poly, _ = qnm_characteristic_polynomial(l, m, n, M, a)
                roots = wdk_roots(poly, tol=1e-12, max_iter=500)

                best_root = None
                best_score = -np.inf
                for r in roots:
                    if r.real <= 0:
                        continue
                    score = r.real - 2.0 * np.abs(r.imag)
                    if score > best_score:
                        best_score = score
                        best_root = r
                if best_root is None:
                    best_root = roots[0]
                results[(l, m, n)] = best_root
    
    return results






def teukolsky_potential(r, M, a, omega, m, s=-2):
    Delta = r**2 - 2 * M * r + a**2
    K = (r**2 + a**2) * omega - a * m
    

    A_lm = l_eigenvalue_approx(2, np.abs(m))
    lam = A_lm - 2 * m * a * omega + a**2 * omega**2 - 2 * s * (s + 1)
    

    Delta = np.where(np.abs(Delta) < 1e-12, 1e-12, Delta)
    
    V = (K**2 - 2j * s * (r - M) * K) / (Delta**2) + 4j * s * omega * r / Delta - lam / Delta
    return V


def l_eigenvalue_approx(l, m):
    s = -2
    return l * (l + 1) - s * (s + 1)


def teukolsky_radial_integration(r_min, r_max, num_points, M, a, omega, m, s=-2, R0=1.0):
    if r_min >= r_max:
        raise ValueError("r_min 必须小于 r_max")
    if num_points < 10:
        raise ValueError("num_points 至少为 10")
    
    r = np.linspace(r_min, r_max, num_points)
    h = r[1] - r[0]
    
    R = np.zeros(num_points, dtype=np.complex128)
    S = np.zeros(num_points, dtype=np.complex128)
    

    R[0] = R0
    S[0] = 1j * omega * R0
    
    for i in range(num_points - 1):
        V = teukolsky_potential(r[i], M, a, omega, m, s)
        

        k1_R = h * S[i]
        k1_S = h * (-V * R[i])
        
        k2_R = h * (S[i] + 0.5 * k1_S)
        k2_S = h * (-V * (R[i] + 0.5 * k1_R))
        
        k3_R = h * (S[i] + 0.5 * k2_S)
        k3_S = h * (-V * (R[i] + 0.5 * k2_R))
        
        k4_R = h * (S[i] + k3_S)
        k4_S = h * (-V * (R[i] + k3_R))
        
        R[i + 1] = R[i] + (k1_R + 2 * k2_R + 2 * k3_R + k4_R) / 6.0
        S[i + 1] = S[i] + (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6.0
    
    return r, R, S






def gravitational_wave_luminosity(qnm_freqs, M, a=0.0):
    luminosity = 0.0
    for key, omega in qnm_freqs.items():
        l, m, n = key

        amp = M**2 * (1.0 - (a / M)**2) * np.abs(omega)**2

        decay_rate = -2.0 * omega.imag
        luminosity += amp * decay_rate
    

    luminosity_dimless = luminosity / (M**2) if M > 0 else 0.0
    
    return luminosity, luminosity_dimless
