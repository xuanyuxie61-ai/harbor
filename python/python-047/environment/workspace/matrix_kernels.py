
import numpy as np
from math import gcd


def r8utt_solve(n, a, b):
    if n <= 0:
        raise ValueError("r8utt_solve: n must be positive")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[0] < n:
        raise ValueError("r8utt_solve: a length {} < n {}".format(a.shape[0], n))
    if abs(a[0]) < 1e-15:
        raise ValueError("r8utt_solve: zero diagonal element a[0]")
    
    if b.ndim == 1:
        x = b.copy()

        for j in range(n - 1, -1, -1):
            x[j] = x[j] / a[0]
            for i in range(j):
                x[i] = x[i] - a[j - i] * x[j]
        return x
    else:

        x = b.copy()
        nrhs = b.shape[1]
        for rhs in range(nrhs):
            for j in range(n - 1, -1, -1):
                x[j, rhs] = x[j, rhs] / a[0]
                for i in range(j):
                    x[i, rhs] = x[i, rhs] - a[j - i] * x[j, rhs]
        return x


def r8utt_to_dense(n, a):
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            A[i, j] = a[j - i]
    return A


def r8utt_inverse(n, a):
    if abs(a[0]) < 1e-15:
        raise ValueError("r8utt_inverse: zero diagonal")
    b = np.zeros(n, dtype=float)
    b[0] = 1.0 / a[0]
    for k in range(1, n):
        s = 0.0
        for j in range(1, k + 1):
            if j < len(a):
                s += a[j] * b[k - j]
        b[k] = -s / a[0]
    return b


def toeplitz_matvec(n, a, x):
    x = np.asarray(x, dtype=float)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i, n):
            if j - i < len(a):
                y[i] += a[j - i] * x[j]
    return y


def invmod_matrix(mat, rmod, cmod):
    mat = np.asarray(mat, dtype=int)
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError("invmod_matrix: mat must be square")
    rmod = np.asarray(rmod, dtype=int).copy()
    cmod = np.asarray(cmod, dtype=int).copy()
    if len(rmod) != n or len(cmod) != n:
        raise ValueError("invmod_matrix: rmod/cmod length mismatch")
    

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if rmod[i] != cmod[j] and val != 0:
                return np.zeros((n, n), dtype=int), 2
            if val < 0 or val >= rmod[i]:
                return np.zeros((n, n), dtype=int), 1
    

    m = mat.copy().reshape(-1)
    imat = np.zeros(n * n, dtype=int)
    

    rsort = np.argsort(rmod)
    csort = np.argsort(cmod)
    rmod_s = rmod[rsort]
    cmod_s = cmod[csort]
    

    mat_s = mat[rsort, :][:, csort].copy().reshape(-1)
    imat_s = np.zeros(n * n, dtype=int)
    

    for idx in range(0, n * n, n + 1):
        imat_s[idx] = 1
    

    for ir in range(n):
        kir = ir * n
        if mat_s[kir + ir] == 0:

            all_zero = True
            kjr_idx = -1
            for jr in range(ir + 1, n):
                if mat_s[jr * n + ir] != 0:
                    all_zero = False
                    kjr_idx = jr
                    break
            
            if all_zero:

                for jr in range(ir):
                    if mat_s[jr * n + ir] != 0:
                        for i in range(jr * n, jr * n + ir):
                            if mat_s[i] != 0:
                                return np.zeros((n, n), dtype=int), 3
                        all_zero = False
                        kjr_idx = jr
                        break
            
            if all_zero:
                continue
            

            kjr = kjr_idx * n
            for i in range(n):
                mat_s[kir + i], mat_s[kjr + i] = mat_s[kjr + i], mat_s[kir + i]
                imat_s[kir + i], imat_s[kjr + i] = imat_s[kjr + i], imat_s[kir + i]
        

        k_val = mat_s[kir + ir]
        mult = -1
        for n_val in range(1, rmod_s[ir]):
            if (n_val * k_val) % rmod_s[ir] == 1:
                mult = n_val
                break
        
        if mult < 0:
            return np.zeros((n, n), dtype=int), 3
        

        if mult > 1:
            for i in range(kir, kir + n):
                mat_s[i] = (mat_s[i] * mult) % cmod_s[i - kir] if (i - kir) < n else mat_s[i] * mult
                imat_s[i] = (imat_s[i] * mult) % cmod_s[i - kir] if (i - kir) < n else imat_s[i] * mult
        

        for kjr_idx in range(n):
            if kjr_idx == ir:
                continue
            kjr = kjr_idx * n
            factor = mat_s[kjr + ir]
            if factor != 0:
                n_sub = (rmod_s[ir] - factor) % rmod_s[ir]
                for i in range(n):
                    cidx = i
                    mod_val = cmod_s[cidx]
                    mat_s[kjr + i] = (mat_s[kjr + i] + n_sub * mat_s[kir + i]) % mod_val
                    imat_s[kjr + i] = (imat_s[kjr + i] + n_sub * imat_s[kir + i]) % mod_val
    

    ifault = 0
    for idx in range(0, n * n, n + 1):
        if mat_s[idx] == 0:
            ifault = -1
    

    for i in range(n):
        for j in range(n):
            if i != j and mat_s[i * n + j] != 0:
                return np.zeros((n, n), dtype=int), 3
    

    imat_mat = imat_s.reshape(n, n)



    inv_rsort = np.argsort(rsort)
    inv_csort = np.argsort(csort)
    imat_final = imat_mat[inv_rsort, :][:, inv_csort]
    
    return imat_final, ifault


def build_toeplitz_green_matrix(nx, ny, nz, dx, dy, dz, obs_z):
    n = nx * ny * nz
    first_row = np.zeros(n, dtype=float)
    
    for idx in range(n):
        k = idx // (nx * ny)
        rem = idx % (nx * ny)
        j = rem // nx
        i = rem % nx
        
        xc = (i - nx // 2) * dx
        yc = (j - ny // 2) * dy
        zc = -k * dz - dz / 2.0
        
        r = np.sqrt(xc**2 + yc**2 + (obs_z - zc)**2)
        r = max(r, 1e-6)
        dV = dx * dy * dz
        first_row[idx] = G_CONST * dV * (obs_z - zc) / (r**3) * 1e5
    
    return first_row


def football_combination_count(max_n):
    if max_n < 0:
        raise ValueError("max_n must be non-negative")
    
    counts = np.zeros(max_n + 1, dtype=np.int64)
    counts[0] = 1
    


    increments = [1, 2, 3, 6, 7, 8]
    
    for n in range(1, max_n + 1):
        total = 0
        for inc in increments:
            if n - inc >= 0:
                total += counts[n - inc]
        counts[n] = total
    
    return counts


def tikhonov_preconditioner_toeplitz(n, green_row, alpha, order=1):



    a = np.zeros(n, dtype=float)
    for lag in range(n):
        s = 0.0
        for k in range(n - lag):
            if k < len(green_row) and k + lag < len(green_row):
                s += green_row[k] * green_row[k + lag]
        a[lag] = s
    a[0] += alpha**2
    

    try:
        inv_row = r8utt_inverse(n, a)
    except ValueError:

        a[0] += 1e-12
        inv_row = r8utt_inverse(n, a)
    
    return inv_row


def sparse_approximate_inverse_mod(nnz_pattern, A_dense, block_size=8):
    A_dense = np.asarray(A_dense, dtype=float)
    n = A_dense.shape[0]
    M_approx = np.zeros((n, n), dtype=float)
    

    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        bs = block_end - block_start
        Ablock = A_dense[block_start:block_end, block_start:block_end]
        

        det = np.linalg.det(Ablock)
        if abs(det) > 1e-12:
            inv_block = np.linalg.inv(Ablock)
        else:

            inv_block = np.linalg.pinv(Ablock)
        

        pattern_block = nnz_pattern[block_start:block_end, block_start:block_end]
        inv_block = inv_block * pattern_block
        
        M_approx[block_start:block_end, block_start:block_end] = inv_block
    
    return M_approx
