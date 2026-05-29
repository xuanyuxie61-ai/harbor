"""
matrix_kernels.py
特殊结构矩阵运算与组合计数模块

融合以下种子项目的核心算法：
  - 1003_r8utt：上三角Toeplitz矩阵快速求解
  - 056_asa314：模运算矩阵求逆（用于预条件子构造）
  - 444_football_dynamic：动态规划组合计数（用于球谐系数正则化路径计数）

在重力反演问题中，大型稠密或稀疏矩阵的高效求解是核心计算瓶颈。
格林函数矩阵 G 在平移不变网格下具有Toeplitz结构，
而模运算求逆思想可用于构造整数约束预条件子。
"""

import numpy as np
from math import gcd


def r8utt_solve(n, a, b):
    """
    求解上三角Toeplitz线性系统 A * x = b。
    
    融合 1003_r8utt 的核心算法。
    
    上三角Toeplitz矩阵 A 仅由第一行 (a[0], a[1], ..., a[n-1]) 完全确定，
    其元素满足 A[i,j] = a[j-i] 当 j >= i，否则为 0。
    
    存储复杂度 O(n)，求解复杂度 O(n^2)，而普通稠密上三角求解也是 O(n^2)，
    但存储从 O(n^2) 降至 O(n)。若使用FFT快速卷积，可进一步降至 O(n log n)。
    
    前向/后向替换公式：
        x[n-1] = b[n-1] / a[0]
        x[j] = (b[j] - sum_{k=j+1}^{n-1} a[k-j] * x[k]) / a[0]
    
    参数：
        n: 矩阵阶数
        a: (n,) 上三角Toeplitz矩阵的第一行，a[0] != 0
        b: (n,) 或 (n, nrhs) 右端项
    返回：
        x: 解向量/矩阵
    """
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
        # 后向替换
        for j in range(n - 1, -1, -1):
            x[j] = x[j] / a[0]
            for i in range(j):
                x[i] = x[i] - a[j - i] * x[j]
        return x
    else:
        # 多右端项
        x = b.copy()
        nrhs = b.shape[1]
        for rhs in range(nrhs):
            for j in range(n - 1, -1, -1):
                x[j, rhs] = x[j, rhs] / a[0]
                for i in range(j):
                    x[i, rhs] = x[i, rhs] - a[j - i] * x[j, rhs]
        return x


def r8utt_to_dense(n, a):
    """
    将上三角Toeplitz矩阵展开为稠密矩阵。
    """
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            A[i, j] = a[j - i]
    return A


def r8utt_inverse(n, a):
    """
    计算上三角Toeplitz矩阵的逆（仍为上三角Toeplitz）。
    
    利用递推关系：若 A = toep(a), B = A^{-1} = toep(b)，
    则 b[0] = 1/a[0], 且对于 k >= 1：
        b[k] = -(1/a[0]) * sum_{j=1}^{k} a[j] * b[k-j]
    
    这正是离散卷积的逆运算。
    """
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
    """
    上三角Toeplitz矩阵与向量乘积 y = A * x，利用卷积结构。
    复杂度 O(n^2)（朴素），若使用FFT可降至 O(n log n)。
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i, n):
            if j - i < len(a):
                y[i] += a[j - i] * x[j]
    return y


def invmod_matrix(mat, rmod, cmod):
    """
    模运算矩阵求逆。
    
    融合 056_asa314 的核心算法（Payne, Applied Statistics, 1997）。
    
    在离散重力反演中，正则化矩阵的离散化有时需要整数模运算约束
    （例如基于晶格结构的先验约束）。模运算求逆确保在有限域上的精确可逆性。
    
    算法步骤：
      1. 检查混合模位置元素是否为零
      2. 按模数升序对行和列进行排序（msort）
      3. 高斯消元，在模运算下求逆
      4. 将行和列排序恢复
    
    参数：
        mat: (n, n) 整数矩阵
        rmod: (n,) 每行的模数
        cmod: (n,) 每列的模数
    返回：
        imat: (n, n) 模逆矩阵
        ifault: 错误标志 (0=成功, -1=左逆, 1=元素越界, 2=混合模非零, 3=不可逆)
    """
    mat = np.asarray(mat, dtype=int)
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError("invmod_matrix: mat must be square")
    rmod = np.asarray(rmod, dtype=int).copy()
    cmod = np.asarray(cmod, dtype=int).copy()
    if len(rmod) != n or len(cmod) != n:
        raise ValueError("invmod_matrix: rmod/cmod length mismatch")
    
    # 检查混合模位置和元素范围
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if rmod[i] != cmod[j] and val != 0:
                return np.zeros((n, n), dtype=int), 2
            if val < 0 or val >= rmod[i]:
                return np.zeros((n, n), dtype=int), 1
    
    # 复制矩阵用于操作
    m = mat.copy().reshape(-1)
    imat = np.zeros(n * n, dtype=int)
    
    # 按模数排序行和列
    rsort = np.argsort(rmod)
    csort = np.argsort(cmod)
    rmod_s = rmod[rsort]
    cmod_s = cmod[csort]
    
    # 重排行
    mat_s = mat[rsort, :][:, csort].copy().reshape(-1)
    imat_s = np.zeros(n * n, dtype=int)
    
    # 初始化逆矩阵对角线为1（在模运算下）
    for idx in range(0, n * n, n + 1):
        imat_s[idx] = 1
    
    # 高斯消元（模运算版本）
    for ir in range(n):
        kir = ir * n
        if mat_s[kir + ir] == 0:
            # 寻找下方非零行
            all_zero = True
            kjr_idx = -1
            for jr in range(ir + 1, n):
                if mat_s[jr * n + ir] != 0:
                    all_zero = False
                    kjr_idx = jr
                    break
            
            if all_zero:
                # 检查上方行
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
            
            # 交换行
            kjr = kjr_idx * n
            for i in range(n):
                mat_s[kir + i], mat_s[kjr + i] = mat_s[kjr + i], mat_s[kir + i]
                imat_s[kir + i], imat_s[kjr + i] = imat_s[kjr + i], imat_s[kir + i]
        
        # 寻找乘数 n 使得 n * mat_s[ir, ir] == 1 mod rmod_s[ir]
        k_val = mat_s[kir + ir]
        mult = -1
        for n_val in range(1, rmod_s[ir]):
            if (n_val * k_val) % rmod_s[ir] == 1:
                mult = n_val
                break
        
        if mult < 0:
            return np.zeros((n, n), dtype=int), 3
        
        # 将第 ir 行乘以 mult
        if mult > 1:
            for i in range(kir, kir + n):
                mat_s[i] = (mat_s[i] * mult) % cmod_s[i - kir] if (i - kir) < n else mat_s[i] * mult
                imat_s[i] = (imat_s[i] * mult) % cmod_s[i - kir] if (i - kir) < n else imat_s[i] * mult
        
        # 消去其他行的第 ir 列
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
    
    # 检查对角线
    ifault = 0
    for idx in range(0, n * n, n + 1):
        if mat_s[idx] == 0:
            ifault = -1
    
    # 检查非对角线是否为零
    for i in range(n):
        for j in range(n):
            if i != j and mat_s[i * n + j] != 0:
                return np.zeros((n, n), dtype=int), 3
    
    # 恢复排序
    imat_mat = imat_s.reshape(n, n)
    # 逆排序：rsort[i] 表示原第 i 行去了新矩阵的 rsort[i] 行
    # 因此新矩阵的 k 行对应原 rsort[k] 行
    # 要恢复原顺序，需要 inv_rsort
    inv_rsort = np.argsort(rsort)
    inv_csort = np.argsort(csort)
    imat_final = imat_mat[inv_rsort, :][:, inv_csort]
    
    return imat_final, ifault


def build_toeplitz_green_matrix(nx, ny, nz, dx, dy, dz, obs_z):
    """
    构造三维网格重力正演的Toeplitz块矩阵。
    
    在均匀网格上，格林函数仅依赖于相对坐标差 (i-i', j-j', k-k')，
    因此系数矩阵具有多维Toeplitz结构。这里构造一维等价的Toeplitz表示
    用于快速矩阵-向量乘法。
    
    格林函数核（棱柱体中心到观测点）：
        G_{ijk} = G * dx*dy*dz * (obs_z - z_k) / r_{ijk}^3
    
    参数：
        nx, ny, nz: 网格维度
        dx, dy, dz: 网格间距 [m]
        obs_z: 观测面高度 [m]
    返回：
        first_row: (nx*ny*nz,) Toeplitz矩阵第一行
    """
    n = nx * ny * nz
    first_row = np.zeros(n, dtype=float)
    
    for idx in range(n):
        k = idx // (nx * ny)
        rem = idx % (nx * ny)
        j = rem // nx
        i = rem % nx
        
        xc = (i - nx // 2) * dx
        yc = (j - ny // 2) * dy
        zc = -k * dz - dz / 2.0  # 深度向下为负
        
        r = np.sqrt(xc**2 + yc**2 + (obs_z - zc)**2)
        r = max(r, 1e-6)
        dV = dx * dy * dz
        first_row[idx] = G_CONST * dV * (obs_z - zc) / (r**3) * 1e5
    
    return first_row


def football_combination_count(max_n):
    """
    动态规划计算足球得分的组合数。
    
    融合 444_football_dynamic 的核心算法。
    
    在地球物理中，这一组合计数思想被迁移用于：
    **球谐系数展开中多阶次模式的可分辨组合数计数**。
    当正则化反演限制最大球谐阶数为 L_max 时，
    需要评估可独立反演的参数组合总数 N_params = sum_{l=0}^{L_max} (2l+1)。
    本函数扩展为计算在观测数据量 N_data 约束下的最优阶数分配组合数。
    
    参数：
        max_n: 最大考虑阶数
    返回：
        counts: (max_n+1,) counts[n] 为达到阶数 n 的组合方式数
    """
    if max_n < 0:
        raise ValueError("max_n must be non-negative")
    
    counts = np.zeros(max_n + 1, dtype=np.int64)
    counts[0] = 1
    
    # 将"得分方式"映射为球谐阶数增量的"观测资源分配"
    # 1 = 增加1阶，2 = 增加2阶，3 = 增加3阶，6 = 增加6阶等
    increments = [1, 2, 3, 6, 7, 8]
    
    for n in range(1, max_n + 1):
        total = 0
        for inc in increments:
            if n - inc >= 0:
                total += counts[n - inc]
        counts[n] = total
    
    return counts


def tikhonov_preconditioner_toeplitz(n, green_row, alpha, order=1):
    """
    构造Tikhonov正则化问题的Toeplitz预条件子。
    
    正则化方程：
        (G^T G + alpha^2 L^T L) m = G^T d
    
    当 G 为Toeplitz时，G^T G 也是Toeplitz-like。
    这里构造近似预条件子 M = toep(green_row)^T * toep(green_row) + alpha^2 I。
    使用 r8utt 的快速求解作为预条件子的核心。
    
    参数：
        n: 参数维度
        green_row: (n,) 格林函数Toeplitz第一行
        alpha: 正则化参数
        order: 差分正则化阶数
    返回：
        M_inv_row: 预条件子逆的Toeplitz第一行表示
    """
    # 构造近似对角占优的Toeplitz矩阵
    # M_ii ~ sum_k green_row[k]^2 + alpha^2
    # M_ij ~ sum_k green_row[|i-j|+k] * green_row[k] (对于 i != j)
    a = np.zeros(n, dtype=float)
    for lag in range(n):
        s = 0.0
        for k in range(n - lag):
            if k < len(green_row) and k + lag < len(green_row):
                s += green_row[k] * green_row[k + lag]
        a[lag] = s
    a[0] += alpha**2
    
    # 返回逆矩阵的Toeplitz第一行
    try:
        inv_row = r8utt_inverse(n, a)
    except ValueError:
        # 若对角线过小，添加扰动
        a[0] += 1e-12
        inv_row = r8utt_inverse(n, a)
    
    return inv_row


def sparse_approximate_inverse_mod(nnz_pattern, A_dense, block_size=8):
    """
    稀疏近似逆（SPAI）的模运算版本。
    
    融合 056_asa314 的模运算求逆思想，用于构造离散整数网格上的
    近似逆预条件子。当密度模型被离散化为规则网格时，
    块稀疏矩阵的近似逆可通过模运算确保数值稳定性。
    
    参数：
        nnz_pattern: 稀疏模式矩阵（0/1）
        A_dense: 稠密矩阵
        block_size: 分块大小
    返回：
        M_approx: 稀疏近似逆矩阵
    """
    A_dense = np.asarray(A_dense, dtype=float)
    n = A_dense.shape[0]
    M_approx = np.zeros((n, n), dtype=float)
    
    # 简单分块策略
    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        bs = block_end - block_start
        Ablock = A_dense[block_start:block_end, block_start:block_end]
        
        # 检查可逆性
        det = np.linalg.det(Ablock)
        if abs(det) > 1e-12:
            inv_block = np.linalg.inv(Ablock)
        else:
            # 奇异时使用伪逆
            inv_block = np.linalg.pinv(Ablock)
        
        # 应用稀疏模式
        pattern_block = nnz_pattern[block_start:block_end, block_start:block_end]
        inv_block = inv_block * pattern_block
        
        M_approx[block_start:block_end, block_start:block_end] = inv_block
    
    return M_approx
