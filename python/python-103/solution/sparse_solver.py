"""
sparse_solver.py
重启GMRES稀疏线性求解器模块（对应种子项目 760_mgmres）

在光纤脉冲传输的隐式时间步进中，需要求解大型稀疏线性系统。
例如，Crank-Nicolson格式的NLSE离散化产生如下系统：

  (I - Δz/2 · D̂) A^{n+1} = (I + Δz/2 · D̂) A^n + N(A^n)

其中D̂为色散算子的离散矩阵（带宽由高阶导数决定），
在频域中该矩阵为对角阵，但在时域高阶有限差分离散下为带状稀疏矩阵。

GMRES（Generalized Minimal RESidual）算法通过Arnoldi过程
在Krylov子空间 K_m(A, r0) = span{r0, A r0, A² r0, ..., A^{m-1} r0}
中最小化残差范数 ‖b - A x‖。

重启机制：当Krylov子空间维数达到mr时，以当前近似解为初值重新开始。

核心公式：
  Arnoldi过程:
    v_1 = r0 / ‖r0‖
    for j = 1,2,...,m
      w = A v_j
      for i = 1,...,j
        h_{ij} = (w, v_i)
        w = w - h_{ij} v_i
      h_{j+1,j} = ‖w‖
      v_{j+1} = w / h_{j+1,j}

  Givens旋转消除H的下对角线，将最小二乘问题化为上三角。
"""

import numpy as np


def mult_givens(c, s, k, g):
    """
    应用Givens旋转到向量g。
    """
    g = g.copy()
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g[k] = g1
    g[k + 1] = g2
    return g


def ax_crs(a, ia, ja, x, n, nz_num):
    """
    CRS（Compressed Row Storage）格式稀疏矩阵-向量乘法 y = A x。

    参数:
        a: ndarray shape (nz_num,), 非零元素值
        ia: ndarray shape (nz_num,), 行索引（0-based）
        ja: ndarray shape (nz_num,), 列索引（0-based）
        x: ndarray shape (n,), 向量
    """
    y = np.zeros(n)
    for k in range(nz_num):
        y[ia[k]] += a[k] * x[ja[k]]
    return y


def mgmres(a, ia, ja, x0, rhs, n, nz_num, itr_max, mr, tol_abs, tol_rel, verbose=False):
    """
    重启GMRES算法求解 A x = rhs。
    （对应种子项目 760_mgmres）

    参数:
        a, ia, ja: CRS格式稀疏矩阵
        x0: ndarray, 初始猜测
        rhs: ndarray, 右端项
        n: int, 矩阵维数
        nz_num: int, 非零元个数
        itr_max: int, 最大外迭代次数
        mr: int, 重启维数
        tol_abs: float, 绝对残差容差
        tol_rel: float, 相对残差容差
        verbose: bool

    返回:
        x: ndarray, 解向量
    """
    if n < 1:
        return x0.copy()
    if mr > n:
        mr = n
    if mr < 1:
        mr = min(10, n)

    delta = 0.001
    x = x0.copy()
    rho_tol = None
    itr_used = 0

    for itr in range(1, itr_max + 1):
        r = rhs - ax_crs(a, ia, ja, x, n, nz_num)
        rho = np.linalg.norm(r)

        if verbose:
            print(f"  ITR = {itr:8d}  Residual = {rho:e}")

        if itr == 1:
            rho_tol = rho * tol_rel

        v = np.zeros((n, mr + 1))
        v[:, 0] = r / rho if rho > 1e-30 else r

        g = np.zeros(mr + 1)
        g[0] = rho

        h = np.zeros((mr + 1, mr))
        c = np.zeros(mr)
        s = np.zeros(mr)

        converged = False
        for k in range(mr):
            k_copy = k
            v[:, k + 1] = ax_crs(a, ia, ja, v[:, k], n, nz_num)
            av = np.linalg.norm(v[:, k + 1])

            for j in range(k + 1):
                h[j, k] = np.dot(v[:, j], v[:, k + 1])
                v[:, k + 1] -= h[j, k] * v[:, j]

            h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            # 重正交化（数值鲁棒性）
            if av + delta * h[k + 1, k] == av:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], v[:, k + 1])
                    h[j, k] += htmp
                    v[:, k + 1] -= htmp * v[:, j]
                h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            if h[k + 1, k] > 1e-15:
                v[:, k + 1] = v[:, k + 1] / h[k + 1, k]

            # Givens旋转
            if k > 0:
                y = h[:k + 2, k].copy()
                for j in range(k):
                    y = mult_givens(c[j], s[j], j, y)
                h[:k + 2, k] = y

            mu = np.sqrt(h[k, k] ** 2 + h[k + 1, k] ** 2)
            if mu < 1e-30:
                mu = 1.0
            c[k] = h[k, k] / mu
            s[k] = -h[k + 1, k] / mu
            h[k, k] = c[k] * h[k, k] - s[k] * h[k + 1, k]
            h[k + 1, k] = 0.0
            g = mult_givens(c[k], s[k], k, g)

            rho = abs(g[k + 1])
            itr_used += 1

            if verbose:
                print(f"  K =   {k + 1:8d}  Residual = {rho:e}")

            if rho <= rho_tol and rho <= tol_abs:
                converged = True
                break

        # 回代求解上三角系统
        # k_solve 为有效迭代次数 (0-based 的列索引最大值)
        k_solve = k_copy - 1
        if converged:
            k_solve = k - 1
        y = np.zeros(k_solve + 1)
        for i in range(k_solve, -1, -1):
            denom = h[i, i]
            if abs(denom) < 1e-30:
                y[i] = 0.0
            else:
                y[i] = (g[i] - np.dot(h[i, i + 1:k_solve + 1], y[i + 1:k_solve + 1])) / denom

        for i in range(n):
            x[i] += np.dot(v[i, :k_solve + 1], y[:k_solve + 1])

        if converged:
            break

    if verbose:
        print(f"\nMGMRES")
        print(f"  Iterations = {itr_used}")
        print(f"  Final residual = {rho:e}")

    return x


def build_dispersion_matrix_crs(n, dt, beta2, beta3, beta4=0.0):
    """
    构建高阶色散算子的时域有限差分离散稀疏矩阵（CRS格式）。

    色散算子:
      D = i(β₂/2)∂²/∂t² - i(β₃/6)∂³/∂t³ + (β₄/24)∂⁴/∂t⁴

    使用中心差分：
      f'' ≈ (f_{i+1} - 2f_i + f_{i-1}) / dt²
      f''' ≈ (f_{i+2} - 2f_{i+1} + 2f_{i-1} - f_{i-2}) / (2 dt³)
      f'''' ≈ (f_{i+2} - 4f_{i+1} + 6f_i - 4f_{i-1} + f_{i-2}) / dt⁴
    """
    if n < 5 or dt <= 0:
        raise ValueError("build_dispersion_matrix_crs: invalid parameters")

    rows = []
    cols = []
    vals = []

    c2 = 1j * beta2 / (2.0 * dt ** 2)
    c3 = -1j * beta3 / (12.0 * dt ** 3)
    c4 = beta4 / (24.0 * dt ** 4) if beta4 != 0.0 else 0.0

    for i in range(n):
        # 二阶差分
        for di, coef in [(-1, c2), (0, -2.0 * c2), (1, c2)]:
            j = i + di
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
                vals.append(coef)

        # 三阶差分
        for di, coef in [(-2, -c3), (-1, 2.0 * c3), (1, -2.0 * c3), (2, c3)]:
            j = i + di
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
                vals.append(coef)

        # 四阶差分
        if beta4 != 0.0:
            for di, coef in [(-2, c4), (-1, -4.0 * c4), (0, 6.0 * c4), (1, -4.0 * c4), (2, c4)]:
                j = i + di
                if 0 <= j < n:
                    rows.append(i)
                    cols.append(j)
                    vals.append(coef)

    # 合并相同位置的元素
    from collections import defaultdict
    merged = defaultdict(float)
    for r, c, v in zip(rows, cols, vals):
        merged[(r, c)] += v

    nz_num = len(merged)
    a = np.zeros(nz_num, dtype=complex)
    ia = np.zeros(nz_num, dtype=int)
    ja = np.zeros(nz_num, dtype=int)

    for idx, ((r, c), v) in enumerate(sorted(merged.items())):
        a[idx] = v
        ia[idx] = r
        ja[idx] = c

    return a, ia, ja, nz_num
