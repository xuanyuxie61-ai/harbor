
import numpy as np


def r83_cr_fa(n, a):
    if n <= 0:
        raise ValueError("n 必须为正整数")
    
    a_cr = np.zeros((3, 2 * n + 1))
    
    if n == 1:
        if abs(a[1, 0]) < 1e-30:
            raise ValueError("对角元素为零，无法分解")
        a_cr[1, 1] = 1.0 / a[1, 0]
        return a_cr
    
    a_cr[0, 1:n] = a[0, 1:n]
    a_cr[1, 1:n + 1] = a[1, 0:n]
    a_cr[2, 1:n] = a[2, 0:n - 1]
    
    il = n
    ipntp = 0
    
    while il > 1:
        ipnt = ipntp
        ipntp = ipntp + il
        inc = il + 1 if il % 2 == 1 else il
        incr = inc // 2
        il = il // 2
        ihaf = ipntp + incr
        ifulp = ipnt + inc + 1
        
        for ilp in range(incr, 0, -1):
            ifulp = ifulp - 2
            iful = ifulp - 1
            ihaf = ihaf - 1
            
            diag_val = a_cr[1, iful]
            if abs(diag_val) < 1e-30:
                diag_val = 1e-30
            
            a_cr[1, iful + 1] = 1.0 / diag_val
            a_cr[2, iful + 1] = a_cr[2, iful] * a_cr[1, iful + 1]
            a_cr[0, ifulp + 1] = a_cr[0, ifulp + 1] * a_cr[1, ifulp + 2]
            a_cr[1, ihaf + 1] = (a_cr[1, ifulp + 1] 
                                 - a_cr[0, iful + 1] * a_cr[2, iful + 1]
                                 - a_cr[0, ifulp + 1] * a_cr[2, ifulp + 1])
            a_cr[3 - 1, ihaf + 1] = -a_cr[3 - 1, ifulp + 1] * a_cr[3 - 1, ifulp + 2]
            a_cr[0, ihaf + 1] = -a_cr[0, ifulp + 1] * a_cr[0, ifulp + 2]
    
    a_cr[1, ipntp + 2] = 1.0 / a_cr[1, ipntp + 2]
    
    return a_cr


def r83_cr_sl(n, a_cr, b):
    if n <= 0:
        raise ValueError("n 必须为正整数")
    
    if n == 1:
        return np.array([a_cr[1, 1] * b[0]])
    
    rhs = np.zeros(2 * n + 1)
    rhs[1] = 0.0
    rhs[2:n + 2] = b[0:n]
    rhs[n + 2:2 * n + 1] = 0.0
    
    il = n
    ndiv = 1
    ipntp = 0
    
    while il > 1:
        ipnt = ipntp
        ipntp = ipntp + il
        il = il // 2
        ndiv = ndiv * 2
        ihaf = ipntp
        
        for iful in range(ipnt + 1, ipntp, 2):
            ihaf = ihaf + 1
            rhs[ihaf + 1] = (rhs[iful + 1] 
                             - a_cr[2, iful] * rhs[iful]
                             - a_cr[0, iful + 1] * rhs[iful + 2])
    
    rhs[ihaf + 1] = rhs[ihaf + 1] * a_cr[1, ihaf + 1]
    ipnt = ipntp
    
    while ipnt > 0:
        ipntp = ipnt
        ndiv = ndiv // 2
        il = n // ndiv
        ipnt = ipnt - il
        ihaf = ipntp
        
        for ifulm in range(ipnt, ipntp, 2):
            iful = ifulm + 1
            ihaf = ihaf + 1
            rhs[iful + 1] = rhs[ihaf + 1]
            rhs[ifulm + 1] = (a_cr[1, ifulm + 1] * 
                              (rhs[ifulm + 1] 
                               - a_cr[2, ifulm] * rhs[ifulm]
                               - a_cr[0, ifulm + 1] * rhs[iful + 1]))
    
    x = rhs[2:n + 2]
    return x


def r8gb_fa(n, ml, mu, a):
    if n <= 0 or ml < 0 or mu < 0 or ml >= n or mu >= n:
        raise ValueError("参数无效")
    
    alu = np.copy(a)
    m = ml + mu + 1
    info = 0
    pivot = np.zeros(n, dtype=int)
    

    j0 = mu + 1
    j1 = min(n, m) - 1
    for jz in range(j0, j1):
        i0 = m - jz
        if i0 <= ml:
            alu[i0:ml + 1, jz] = 0.0
    
    jz = j1
    ju = 0
    
    for k in range(n - 1):
        jz = jz + 1
        if jz < n:
            alu[0:ml, jz] = 0.0
        
        lm = min(ml, n - k - 1)
        l = m - 1
        
        for j in range(m, m + lm):
            if abs(alu[l, k]) < abs(alu[j, k]):
                l = j
        
        pivot[k] = l + k - m + 1
        
        if abs(alu[l, k]) < 1e-30:
            info = k + 1
            pivot[k] = k
            continue
        

        if l != m - 1:
            t = alu[l, k]
            alu[l, k] = alu[m - 1, k]
            alu[m - 1, k] = t
        

        if abs(alu[m - 1, k]) > 1e-30:
            alu[m:m + lm, k] = -alu[m:m + lm, k] / alu[m - 1, k]
        
        ju = min(max(ju, mu + pivot[k]), n - 1)
        mm = m - 1
        
        for j in range(k + 1, ju + 1):
            l = l - 1
            mm = mm - 1
            
            if l != mm and mm >= 0 and l >= 0:
                t = alu[l, j]
                alu[l, j] = alu[mm, j]
                alu[mm, j] = t
            
            if mm >= 0 and m - 1 >= 0:
                update = alu[mm, j] * alu[m - 1:m + lm - 1, k]

                update = np.clip(update, -1e200, 1e200)
                alu[mm + 1:mm + lm + 1, j] = alu[mm + 1:mm + lm + 1, j] + update
    
    pivot[n - 1] = n - 1
    if abs(alu[m - 1, n - 1]) < 1e-30:
        info = n
    
    return alu, pivot, info


def r8gb_sl(n, ml, mu, alu, pivot, b, job=0):
    if n <= 0:
        raise ValueError("n 必须为正")
    
    x = np.copy(b).astype(float)
    m = ml + mu + 1
    
    if job == 0:

        for k in range(n - 1):
            l = int(pivot[k])
            if l < 0 or l >= n:
                l = k
            t = float(x[l])
            
            if l != k:
                x[l] = float(x[k])
                x[k] = t
            
            lm = min(ml, n - k - 1)
            if abs(t) > 1e-30 and lm > 0 and m < alu.shape[0]:
                for idx in range(k + 1, k + lm + 1):
                    alu_idx = m + (idx - k - 1)
                    if alu_idx < alu.shape[0] and idx < n:

                        add_val = t * float(alu[alu_idx, k])
                        if np.isfinite(add_val):
                            x[idx] = float(x[idx]) + add_val
        

        for k in range(n - 1, -1, -1):
            diag_val = float(alu[m - 1, k])
            if abs(diag_val) > 1e-30 and np.isfinite(diag_val):
                x[k] = float(x[k]) / diag_val
            else:
                x[k] = 0.0
            
            t = -float(x[k])
            start = max(0, k - mu)
            if abs(t) > 1e-30 and start < k:
                for idx in range(start, k):
                    alu_idx = m - 1 - (k - idx)
                    if alu_idx >= 0 and alu_idx < alu.shape[0]:
                        add_val = t * float(alu[alu_idx, k])
                        if np.isfinite(add_val):
                            x[idx] = float(x[idx]) + add_val
    else:

        for k in range(n):
            t = 0.0
            start = max(0, k - mu)
            if start < k and m - 2 >= 0:
                for idx in range(start, k):
                    alu_idx = m - 1 - (k - idx)
                    if alu_idx >= 0 and alu_idx < alu.shape[0]:
                        t += float(alu[alu_idx, k]) * float(x[idx])
            diag_val = float(alu[m - 1, k])
            if abs(diag_val) > 1e-30:
                x[k] = (float(x[k]) - t) / diag_val
            else:
                x[k] = 0.0
        
        for k in range(n - 2, -1, -1):
            lm = min(ml, n - k - 1)
            if m - 1 >= 0 and lm > 0:
                for idx in range(k + 1, k + lm + 1):
                    alu_idx = m + (idx - k - 1)
                    if alu_idx < alu.shape[0] and idx < n:
                        x[k] += float(alu[alu_idx, k]) * float(x[idx])
            l = int(pivot[k])
            if l != k and 0 <= l < n:
                t = float(x[l])
                x[l] = float(x[k])
                x[k] = t
    

    x = np.array([float(v) if np.isfinite(v) else 0.0 for v in x])
    return x


def thomas_algorithm(lower, diag, upper, rhs):
    n = len(diag)
    if n <= 0:
        return np.array([])
    

    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    

    c_prime[0] = upper[0] / diag[0] if abs(diag[0]) > 1e-30 else 0.0
    d_prime[0] = rhs[0] / diag[0] if abs(diag[0]) > 1e-30 else 0.0
    
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom
    

    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    
    return x


def solve_diffusion_tridiagonal(D_eff, k_rxn, L_ccl, C_0, N=100):
    if D_eff <= 0 or k_rxn < 0 or L_ccl <= 0 or C_0 < 0 or N < 2:
        raise ValueError("物理参数必须满足: D_eff>0, k_rxn>=0, L_ccl>0, C_0>=0, N>=2")
    
    dx = L_ccl / (N - 1)
    x = np.linspace(0, L_ccl, N)
    


    





    raise NotImplementedError("Hole_2: 请实现 solve_diffusion_tridiagonal 的三对角系统构建")
    

    C = np.clip(C, 0.0, C_0 * 1.5)
    
    return x, C


def solve_diffusion_banded(D_eff, k_rxn, L_ccl, C_0, N=100):
    dx = L_ccl / max(N - 1, 1)
    x = np.linspace(0, L_ccl, N)
    
    coeff = D_eff / (dx * dx)
    ml, mu = 1, 1
    m_band = 2 * ml + mu + 1
    
    a_band = np.zeros((m_band, N))
    b_vec = np.zeros(N)
    

    for i in range(1, N - 1):
        a_band[ml + mu, i] = 2.0 * coeff + k_rxn
        a_band[ml + mu - 1, i] = -coeff
        a_band[ml + mu + 1, i - 1] = -coeff
    

    a_band[ml + mu, 0] = 1.0
    b_vec[0] = C_0
    a_band[ml + mu, N - 1] = 1.0
    a_band[ml + mu - 1, N - 1] = -1.0
    b_vec[N - 1] = 0.0
    
    alu, pivot, info = r8gb_fa(N, ml, mu, a_band)
    if info != 0:

        C = np.linspace(C_0, C_0, N)
    else:
        C = r8gb_sl(N, ml, mu, alu, pivot, b_vec)
    
    C = np.clip(C, 0.0, C_0 * 1.5)
    return x, C


def effective_diffusivity(D_bulk, epsilon, tau=1.5):
    if epsilon < 0 or epsilon > 1:
        raise ValueError("孔隙率 epsilon 必须在 [0,1] 之间")
    if D_bulk <= 0:
        raise ValueError("体相扩散系数必须为正")
    
    D_eff = D_bulk * (epsilon ** tau)
    return max(D_eff, 1e-15)


if __name__ == "__main__":
    D_bulk = 2.1e-5
    eps = 0.4
    D_eff = effective_diffusivity(D_bulk, eps)
    x, C = solve_diffusion_tridiagonal(D_eff, 100.0, 10e-6, 1.2, N=51)
    print(f"CCL 厚度 10 um, C(0)={C[0]:.4f}, C(L)={C[-1]:.4f} mol/m^3")
