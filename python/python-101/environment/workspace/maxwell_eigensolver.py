
import numpy as np
from physics_core import C_0, helmholtz_operator_2d_te






def r8pp_fa(n, a):
    if n < 1:
        raise ValueError("矩阵阶数必须 >= 1")
    expected_len = n * (n + 1) // 2
    if len(a) != expected_len:
        raise ValueError(f"packed 数组长度应为 {expected_len}，实际为 {len(a)}")
    
    r = a.copy()
    info = 0
    
    for j in range(n):
        s = 0.0
        for k in range(j):
            idx_kj = k + j * (j + 1) // 2
            t = r[idx_kj]
            for i in range(k):
                idx_ik = i + k * (k + 1) // 2
                idx_ij = i + j * (j + 1) // 2
                t -= r[idx_ik] * r[idx_ij]
            idx_kk = k + k * (k + 1) // 2
            if abs(r[idx_kk]) < 1e-15:
                info = k + 1
                return r, info
            t /= r[idx_kk]
            r[idx_kj] = t
            s += t * t
        
        idx_jj = j + j * (j + 1) // 2
        s = r[idx_jj] - s
        
        if s <= 0.0:
            info = j + 1
            return r, info
        
        r[idx_jj] = np.sqrt(s)
    
    return r, info


def r8pp_sl(n, r_factor, b):
    if n < 1:
        raise ValueError("矩阵阶数必须 >= 1")
    b = np.asarray(b, dtype=float)
    if b.shape != (n,):
        raise ValueError("b 的形状必须与 n 一致")
    
    x = b.copy()
    

    for k in range(n):
        t = 0.0
        for i in range(k):
            idx_ik = i + k * (k + 1) // 2
            t += r_factor[idx_ik] * x[i]
        idx_kk = k + k * (k + 1) // 2
        if abs(r_factor[idx_kk]) < 1e-15:
            x[k] = 0.0
        else:
            x[k] = (x[k] - t) / r_factor[idx_kk]
    

    for k in range(n - 1, -1, -1):
        idx_kk = k + k * (k + 1) // 2
        if abs(r_factor[idx_kk]) < 1e-15:
            x[k] = 0.0
        else:
            x[k] /= r_factor[idx_kk]
        t = -x[k]
        for i in range(k):
            idx_ik = i + k * (k + 1) // 2
            x[i] += t * r_factor[idx_ik]
    
    return x


def r8pp_mv(n, a, x_vec):
    x_vec = np.asarray(x_vec, dtype=float)
    b = np.zeros(n, dtype=float)
    
    for i in range(n):
        for j in range(i):
            k = j + (i * (i + 1)) // 2
            b[i] += a[k] * x_vec[j]
        for j in range(i, n):
            k = i + (j * (j + 1)) // 2
            b[i] += a[k] * x_vec[j]
    
    return b






def gauss_seidel_step(n, A, b, x):
    x_new = np.zeros(n, dtype=float)
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            raise ValueError(f"对角元 A[{i},{i}] 接近零，GS 迭代不收敛")
        x_new[i] = b[i]
        x_new[i] -= np.dot(A[i, :i], x_new[:i])
        x_new[i] -= np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] /= A[i, i]
    return x_new


def gauss_seidel_solve(A, b, x0=None, tol=1e-10, max_iter=10000):
    n = len(b)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        b_norm = 1.0
    
    residual_history = []
    for it in range(max_iter):
        x = gauss_seidel_step(n, A, b, x)
        res = np.linalg.norm(A.dot(x) - b) / b_norm
        residual_history.append(res)
        if res < tol:
            break
    
    return x, residual_history






def st_to_ccs_size(nst, ist, jst):
    if nst == 0:
        return 0
    pairs = set()
    for k in range(nst):
        pairs.add((ist[k], jst[k]))
    return len(pairs)


def st_to_ccs_index(nst, ist, jst, ncc, n):
    if nst == 0:
        return np.array([], dtype=int), np.zeros(n + 1, dtype=int)
    

    data = []
    for k in range(nst):
        data.append((jst[k], ist[k]))
    data = sorted(set(data))
    
    icc = np.array([row for _, row in data], dtype=int)
    jcc = np.array([col for col, _ in data], dtype=int)
    
    ccc = np.zeros(n + 1, dtype=int)
    ccc[0] = 0
    jlo = 0
    for i in range(ncc):
        jhi = jcc[i]
        if jhi != jlo:
            ccc[jlo + 1:jhi + 1] = i
            jlo = jhi
    ccc[jlo + 1:] = ncc
    
    return icc, ccc


def st_to_ccs_values(nst, ist, jst, ast, ncc, n, icc, ccc):
    acc = np.zeros(ncc, dtype=float)
    
    for kst in range(nst):
        i = ist[kst]
        j = jst[kst]
        clo = ccc[j]
        chi = ccc[j + 1]
        
        found = False
        for kcc in range(clo, chi):
            if icc[kcc] == i:
                acc[kcc] += ast[kst]
                found = True
                break
        
        if not found:
            raise ValueError(f"ST 条目 ({i},{j}) 无法在 CCS 数组中定位")
    
    return acc


def ccs_mv(n, icc, ccc, acc, x_vec):
    b = np.zeros(n, dtype=float)
    for j in range(n):
        for kcc in range(ccc[j], ccc[j + 1]):
            i = icc[kcc]
            b[i] += acc[kcc] * x_vec[j]
    return b






def sine_transform_data(n, f_vals):
    f_vals = np.asarray(f_vals, dtype=float)
    if len(f_vals) != n:
        raise ValueError("f_vals 长度必须等于 n")
    
    s = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            s[i] += np.sin(np.pi * (i + 1) * (j + 1) / (n + 1)) * f_vals[j]
    s *= np.sqrt(2.0 / (n + 1))
    return s


def sine_transform_interpolant(n, a, b, s_coeffs, x_query):
    if not (a <= x_query <= b):
        raise ValueError("查询点必须在 [a, b] 区间内")
    if abs(b - a) < 1e-15:
        raise ValueError("区间长度必须为正")
    

    norm = np.sqrt(2.0 / (n + 1))
    f_interp = 0.0
    for k in range(n):
        f_interp += s_coeffs[k] * np.sin(np.pi * (k + 1) * (x_query - a) / (b - a)) * norm
    
    return f_interp






def build_pwe_matrix(n_g, eps_r, a, kx, ky):
    nx, ny = eps_r.shape
    N_pw = (2 * n_g + 1) ** 2
    

    G_vec = []
    for i in range(-n_g, n_g + 1):
        for j in range(-n_g, n_g + 1):
            G_vec.append([i * 2 * np.pi / a, j * 2 * np.pi / a])
    G_vec = np.array(G_vec)
    

    inv_eps = 1.0 / np.maximum(eps_r, 1e-12)
    kappa_G = np.fft.fft2(inv_eps) / (nx * ny)
    
    H = np.zeros((N_pw, N_pw), dtype=complex)
    
    for m in range(N_pw):
        for n in range(N_pw):
            dG = G_vec[m] - G_vec[n]

            ig = int(np.round(dG[0] * a / (2 * np.pi))) % nx
            jg = int(np.round(dG[1] * a / (2 * np.pi))) % ny
            kappa = kappa_G[ig, jg]
            
            k_plus_G = np.array([kx, ky]) + G_vec[n]
            k_mag2 = k_plus_G[0] ** 2 + k_plus_G[1] ** 2
            
            H[m, n] = kappa * k_mag2
    
    return H, G_vec


def solve_bands_pwe(n_bands, n_g, eps_r, a, k_points):
    N_k = len(k_points)
    omega_bands = np.zeros((N_k, n_bands))
    
    for ik, (kx, ky) in enumerate(k_points):
        H, _ = build_pwe_matrix(n_g, eps_r, a, kx, ky)

        H_sym = 0.5 * (H + H.conj().T)
        

        H_sym += 1e-14 * np.eye(H_sym.shape[0])
        
        eigenvalues = np.linalg.eigvalsh(H_sym)
        eigenvalues = np.sort(np.real(eigenvalues))
        

        for ib in range(min(n_bands, len(eigenvalues))):
            lam = max(eigenvalues[ib], 0.0)
            omega_bands[ik, ib] = C_0 * np.sqrt(lam)
    
    return omega_bands






def solve_layered_structure_spectral(n_modes, n_pts, eps_profile, a, kx):
    if n_modes > n_pts:
        raise ValueError("模态数不能超过空间点数")
    
    dx = a / (n_pts + 1)
    x = np.linspace(dx, a - dx, n_pts)
    

    K = np.zeros((n_modes, n_modes), dtype=float)
    
    for n in range(n_modes):
        for m in range(n_modes):

            integrand = np.zeros(n_pts)
            for i in range(n_pts):
                sin_n = np.sin((n + 1) * np.pi * x[i] / a)
                sin_m = np.sin((m + 1) * np.pi * x[i] / a)

                laplacian_m = ((m + 1) * np.pi / a) ** 2 * sin_m + kx ** 2 * sin_m
                integrand[i] = sin_n * (1.0 / max(eps_profile[i], 1e-12)) * laplacian_m
            
            K[n, m] = (2.0 / a) * np.trapz(integrand, x)
    

    K = 0.5 * (K + K.T)
    
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
    
    omega = C_0 * np.sqrt(eigenvalues)
    modes = eigenvectors.T
    
    return omega, modes
