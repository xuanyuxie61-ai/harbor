"""
sparse_grid_collocation.py — 稀疏网格随机配置法
================================================

核心方法:
  1. 张量积超立方体网格 (映射自 558_hypercube_grid)
  2. Van der Corput 拟随机序列 QMC (映射自 1380_van_der_corput)
  3. Ulam 螺旋索引映射稀疏网格自适应 (映射自 1371_ulam_spiral)
  4. Smolyak 稀疏网格配置法

Smolyak 算子:
  A(q,d) = Σ (-1)^(q-|i|) C(d-1,q-|i|) (Q^{i₁}⊗...⊗Q^{i_d})
  其中 q-d+1 ≤ |i| ≤ q.

映射种子项目:
  - 558_hypercube_grid: 张量积网格直接乘积构造
  - 1380_van_der_corput: 低差异拟随机序列
  - 1371_ulam_spiral: 螺旋枚举多指标集
"""

import numpy as np
from itertools import product as iter_product
from scipy.special import roots_hermite, roots_legendre


# ============================================================
# 第1部分: 张量积超立方体网格
# (映射自 558_hypercube_grid: hypercube_grid + r8vec_direct_product)
# ============================================================

def direct_product_1d(factor_index, factor_order, factor_value, factor_num, point_num):
    """
    直接乘积算法: 将一维网格值扩展到高维张量积网格。
    stride pattern: contig=Π_{k<i} ns[k], rep=Π_{k>i} ns[k], skip=contig*order
    """
    x = np.zeros(point_num)
    contig = factor_order ** factor_index
    rep = factor_order ** (factor_num - factor_index - 1)
    idx = 0
    for r in range(rep):
        for v in range(factor_order):
            for c in range(contig):
                if idx < point_num:
                    x[idx] = factor_value[v]
                    idx += 1
    return x


def hypercube_grid_tensor(m, ns, a, b, centering='endpoint'):
    """
    m 维超立方体 [a₁,b₁]×...×[a_m,b_m] 上张量积网格。
    支持 5 种 centering: endpoint/interior/centered/half_left/half_right
    返回: grid (m, N), N = Π ns[i]
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ns = np.asarray(ns, dtype=int)
    grids_1d = []
    for dim in range(m):
        n = ns[dim]
        if centering == 'endpoint':
            pts = np.linspace(a[dim], b[dim], n)
        elif centering == 'interior':
            all_pts = np.linspace(a[dim], b[dim], n + 2)
            pts = all_pts[1:-1]
        elif centering == 'centered':
            h = (b[dim] - a[dim]) / n
            pts = a[dim] + h * (np.arange(n) + 0.5)
        elif centering == 'half_left':
            pts = np.linspace(a[dim], b[dim], n + 1)[:-1]
        elif centering == 'half_right':
            pts = np.linspace(a[dim], b[dim], n + 1)[1:]
        else:
            pts = np.linspace(a[dim], b[dim], n)
        grids_1d.append(pts)
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    total_points = int(np.prod(ns))
    grid = np.zeros((m, total_points))
    for dim in range(m):
        grid[dim, :] = mesh[dim].ravel()
    return grid


def compute_quadrature_weights_1d(points, rule='gauss-legendre'):
    """
    一维正交配置权重:
      gauss-legendre: 精度 2n-1
      gauss-hermite:  权重 e^{-x²}
      clenshaw-curtis: 余弦节点
      trapezoidal:    梯形法则
    """
    n = len(points)
    if n <= 0:
        return np.array([])
    if rule == 'gauss-legendre':
        ref_pts, ref_wts = roots_legendre(n)
        a, b = points[0], points[-1]
        scale = (b - a) / 2.0
        return ref_wts * scale
    elif rule == 'gauss-hermite':
        _, ref_wts = roots_hermite(n)
        return ref_wts
    elif rule == 'clenshaw-curtis':
        if n == 1:
            return np.array([points[-1] - points[0]])
        weights = np.zeros(n)
        theta = np.pi * np.arange(n) / (n - 1)
        for j in range(n):
            s = 0.0
            for k in range(1, (n - 1) // 2 + 1):
                b_k = 2.0 if 2 * k < n - 1 else 1.0
                s += b_k * np.cos(2.0 * k * theta[j]) / (4.0 * k ** 2 - 1.0)
            weights[j] = (1.0 - s) * (points[-1] - points[0]) / (n - 1)
        weights[0] /= 2.0
        weights[-1] /= 2.0
        return weights
    else:
        weights = np.zeros(n)
        for i in range(n - 1):
            h = points[i + 1] - points[i]
            weights[i] += h / 2.0
            weights[i + 1] += h / 2.0
        return weights


# ============================================================
# 第2部分: Van der Corput 拟随机序列
# (映射自 1380_van_der_corput)
# ============================================================

def van_der_corput_sequence(n, base=2):
    """
    Van der Corput 序列 ∈ [0,1]:
        key → d_k...d_0 (base) → 0.d_0...d_k (base) = Σ dᵢ·base^{-(i+1)}
    差异: D_N = O(log N / N) 优于随机 O(N^{-1/2})
    """
    seq = np.zeros(n)
    for key in range(1, n + 1):
        val = 0.0
        base_inv = 1.0 / base
        k = key
        while k > 0:
            digit = k % base
            val += digit * base_inv
            k //= base
            base_inv /= base
        seq[key - 1] = val
    return seq


def halton_sequence(n, dimensions):
    """
    Halton 序列: Van der Corput 的多维推广, 每维用不同素数基数。
    返回: points (n, dimensions)
    """
    primes = _generate_primes(dimensions)
    points = np.zeros((n, dimensions))
    for d in range(dimensions):
        points[:, d] = van_der_corput_sequence(n, base=primes[d])
    return points


def _generate_primes(n):
    """生成前 n 个素数"""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in range(2, int(candidate ** 0.5) + 1))
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


# ============================================================
# 第3部分: Ulam 螺旋索引映射
# (映射自 1371_ulam_spiral)
# ============================================================

def ulam_spiral_index(thickness):
    """
    Ulam 螺旋阵列: 整数 1,...,(2t+1)² 的方形螺旋排列。
    用于稀疏网格多指标集的自适应层级枚举。
    """
    size = 2 * thickness + 1
    spiral = np.zeros((size, size), dtype=int)
    x, y = thickness, thickness
    spiral[y, x] = 1
    counter = 2
    for layer in range(1, thickness + 1):
        x += 1
        if 0 <= y < size and 0 <= x < size:
            spiral[y, x] = counter
            counter += 1
        for _ in range(2 * layer - 1):
            y -= 1
            if 0 <= y < size and 0 <= x < size:
                spiral[y, x] = counter
                counter += 1
        for _ in range(2 * layer):
            x -= 1
            if 0 <= y < size and 0 <= x < size:
                spiral[y, x] = counter
                counter += 1
        for _ in range(2 * layer):
            y += 1
            if 0 <= y < size and 0 <= x < size:
                spiral[y, x] = counter
                counter += 1
        for _ in range(2 * layer):
            x += 1
            if 0 <= y < size and 0 <= x < size:
                spiral[y, x] = counter
                counter += 1
    return spiral


def spiral_to_multiindex(spiral_val, max_level):
    """螺旋索引 → 稀疏网格多指标层级映射"""
    size = 2 * max_level + 1
    center = max_level
    idx_array = np.arange(1, size * size + 1).reshape(size, size)
    coords = np.argwhere(spiral_val == idx_array)
    if len(coords) == 0:
        return (0, 0)
    y_off = coords[0][0] - center
    x_off = coords[0][1] - center
    return (abs(y_off), abs(x_off))


# ============================================================
# 第4部分: Smolyak 稀疏网格构造
# ============================================================

def smolyak_multiindices(d, q):
    """
    Smolyak 多指标集:
        I(q,d) = {i ∈ ℕ^d : q-d+1 ≤ |i| ≤ q, i_k ≥ 1}
    """
    indices = []
    lower = max(1, q - d + 1)
    for total in range(lower, q + 1):
        for combo in _compositions(total, d, min_val=1):
            indices.append(tuple(combo))
    return indices


def _compositions(n, k, min_val=1):
    """n 的 k 部分组合, 每部分 ≥ min_val"""
    if k == 1:
        if n >= min_val:
            yield [n]
        return
    for i in range(min_val, n - min_val * (k - 1) + 1):
        for rest in _compositions(n - i, k - 1, min_val):
            yield [i] + rest


def build_smolyak_grid(d, q, quadrature_func=None):
    """
    Smolyak 稀疏网格:
      对每个 i ∈ I(q,d): X_i = X^{i₁}×...×X^{i_d}
      A(q,d)·f = Σ (-1)^{q-|i|} C(d-1,q-|i|) (Q^{i₁}⊗...⊗Q^{i_d})·f
    """
    if quadrature_func is None:
        def quadrature_func(level):
            pts, wts = roots_legendre(max(1, level))
            return pts, wts
    multi_indices = smolyak_multiindices(d, q)
    all_points = []
    all_weights = []
    from math import comb
    for mi in multi_indices:
        pts_1d, wts_1d = [], []
        for dim_idx in range(d):
            p, w = quadrature_func(mi[dim_idx])
            pts_1d.append(p)
            wts_1d.append(w)
        mesh_pts = np.array(list(iter_product(*pts_1d)))
        if mesh_pts.ndim == 1:
            mesh_pts = mesh_pts.reshape(-1, 1)
        mesh_wts = np.ones(len(mesh_pts))
        for dim_idx in range(d):
            n_after = int(np.prod([len(pts_1d[k]) for k in range(dim_idx + 1, d)]))
            n_before = int(np.prod([len(pts_1d[k]) for k in range(dim_idx)]))
            w_rep = np.repeat(wts_1d[dim_idx], max(1, n_after))
            w_til = np.tile(w_rep, max(1, n_before))
            mesh_wts *= w_til
        total_level = sum(mi)
        sign = (-1) ** (q - total_level)
        binom_coeff = comb(d - 1, q - total_level)
        coeff = sign * binom_coeff
        all_points.append(mesh_pts)
        all_weights.append(coeff * mesh_wts)
    if len(all_points) == 0:
        return np.zeros((0, d)), np.array([])
    points = np.vstack(all_points)
    weights = np.concatenate(all_weights)
    return points, weights


# ============================================================
# 第5部分: 自适应误差估计
# ============================================================

def estimate_sparse_grid_error(function_values, weights):
    """
    稀疏网格误差估计 (层级差分):
        ε ≈ ||A(q,d)·f - A(q-1,d)·f||
    """
    if len(function_values) == 0:
        return 0.0
    weighted_mean = np.abs(np.sum(function_values * weights))
    weighted_var = np.sum(weights ** 2 * function_values ** 2)
    return np.sqrt(max(0.0, weighted_var - weighted_mean ** 2))


def qmc_estimator(n_samples, dimensions, seed=None):
    """Halton 序列 QMC 估计器 ∈ [0,1]^d"""
    return halton_sequence(n_samples, dimensions)
