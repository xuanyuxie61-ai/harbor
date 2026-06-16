"""
cube_exactness_quadrature.py — 高维求积精确度检验与随机期望计算

科学背景
========
在 UQ 中需要计算随机期望:
    E[g(u(·,ω))] = ∫ g(u(·,ξ)) ρ(ξ) dξ

其中 ρ(ξ) 为标准正态密度, ξ ∈ ℝ^K (KL 维度).

Gauss-Chebyshev 第一类求积规则:
    ∫_{-1}^{1} f(x) / √(1-x²) dx ≈ Σ_{k=1}^{n} w_k · f(x_k)
    x_k = cos((2k-1)π/(2n)),  w_k = π/n

精确度检验: 规则对 2n-1 次多项式精确.

算法来源 (种子项目 231_cube_exactness, 164_chebyshev1_exactness)
================================================================
种子 231: 3D Legendre 求积的单项式精确度测试
种子 164: 1D Chebyshev 第一类求积的精确度测试

在本项目中的角色
================
1. 计算随机期望 E[u(x,t)] 和 Var[u(x,t)]
2. 验证求积规则对多项式测试函数的精确度
3. 计算置信带覆盖概率的数值积分

核心公式
========
1. 3D Legendre 单项式积分:
   I(i,j,k) = ∫_{-1}^{1}∫_{-1}^{1}∫_{-1}^{1} x^i y^j z^k dx dy dz
   = [x^{i+1}/(i+1)]_{-1}^{1} · [y^{j+1}/(j+1)]_{-1}^{1} · [z^{k+1}/(k+1)]_{-1}^{1}
2. Gauss-Hermite 求积 (正态权重):
   ∫ f(x)·φ(x) dx ≈ Σ w_k · f(x_k)
3. 张量积求积:
   ∫...∫ f(ξ)·φ(ξ) dξ ≈ Σ_{k1,...,kK} w_{k1}...w_{kK} · f(ξ_{k1},...,ξ_{kK})
"""

import numpy as np


def gauss_hermite_nodes_weights(n):
    """Gauss-Hermite 求积节点和权重 (概率论形式, 正态权重).

    ∫_{-∞}^{∞} f(x) · (1/√(2π))·exp(-x²/2) dx
        ≈ Σ_{k=1}^{n} w_k · f(x_k)

    使用 Golub-Welsch 算法求解概率论 Hermite 多项式的根.

    参数
    ----
    n : int  求积阶数

    返回
    ----
    nodes : ndarray, shape (n,)
    weights : ndarray, shape (n,)  和为 1
    """
    if n <= 0:
        return np.array([0.0]), np.array([1.0])
    if n == 1:
        return np.array([0.0]), np.array([1.0])

    # Golub-Welsch: 概率论 Hermite 多项式的三对角 Jacobi 矩阵
    # 递推: x·He_n = He_{n+1} + n·He_{n-1}
    # 对应 β_k = √k (k=1,...,n-1)
    i = np.arange(1, n, dtype=float)
    beta = np.sqrt(i)  # √1, √2, ..., √(n-1)
    J = np.diag(beta, -1) + np.diag(beta, 1)

    eigenvalues, eigenvectors = np.linalg.eigh(J)

    nodes = eigenvalues
    # 权重: w_k = √π · v_k[0]²  (物理学家 Hermite)
    # 对概率论形式: 权重已经对应正态测度, 归一化即可
    weights = eigenvectors[0, :] ** 2
    weights = weights / np.sum(weights)  # 归一化使和为 1

    return nodes, weights


def legendre_1d_monomial_integral(a, b, p):
    """1D Legendre 单项式积分.

    ∫_a^b x^p dx = [x^{p+1}/(p+1)]_a^b

    参数
    ----
    a, b : float
    p : int

    返回
    ----
    value : float
    """
    if p == -1:
        return np.log(b / a) if a > 0 else 0.0
    return (b ** (p + 1) - a ** (p + 1)) / (p + 1)


def legendre_3d_monomial_integral(a, b, p):
    """3D Legendre 单项式积分 (种子 231).

    I(p) = Π_{d=1}^{3} ∫_{a_d}^{b_d} x_d^{p_d} dx_d

    参数
    ----
    a, b : ndarray, shape (3,)
    p : ndarray, shape (3,)  各维度的幂次

    返回
    ----
    value : float
    """
    value = 1.0
    for d in range(3):
        value *= legendre_1d_monomial_integral(a[d], b[d], int(p[d]))
    return value


def chebyshev1_exactness_test(n_quad, degree_max):
    """测试 Chebyshev 第一类求积的单项式精确度 (种子 164).

    ∫_{-1}^{1} x^p / √(1-x²) dx = π  (p 偶), 0 (p 奇)

    参数
    ----
    n_quad : int  求积点数
    degree_max : int  最高测试次数

    返回
    ----
    errors : dict {degree: relative_error}
    """
    k = np.arange(1, n_quad + 1)
    nodes = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n_quad))
    weights = np.full(n_quad, np.pi / n_quad)

    errors = {}
    for p in range(degree_max + 1):
        # 精确值
        if p % 2 == 1:
            exact = 0.0
        else:
            # ∫_{-1}^{1} x^p / √(1-x²) dx = B((p+1)/2, 1/2)
            from scipy.special import beta as beta_func
            exact = beta_func((p + 1) / 2.0, 0.5)

        # 数值近似
        approx = np.sum(weights * nodes ** p)

        if abs(exact) < 1e-30:
            errors[p] = abs(approx)
        else:
            errors[p] = abs(approx - exact) / abs(exact)

    return errors


def legendre_3d_exactness_test(n_per_dim, degree_max):
    """测试 3D Legendre 求积的精确度 (种子 231).

    参数
    ----
    n_per_dim : int  每维求积点数
    degree_max : int

    返回
    ----
    errors : list of dict
    """
    nodes_1d, weights_1d = gauss_hermite_nodes_weights(n_per_dim)
    # 映射到 [-1, 1]
    nodes_1d = nodes_1d / np.max(np.abs(nodes_1d)) if np.max(np.abs(nodes_1d)) > 0 else nodes_1d
    weights_1d = weights_1d / np.sum(weights_1d) * 2.0

    a = np.array([-1.0, -1.0, -1.0])
    b = np.array([1.0, 1.0, 1.0])

    errors = []
    for total_deg in range(degree_max + 1):
        for pk in range(total_deg + 1):
            for pj in range(total_deg - pk + 1):
                pi = total_deg - pk - pj
                p = np.array([pi, pj, pk])

                exact = legendre_3d_monomial_integral(a, b, p)

                # 张量积求积
                approx = 0.0
                for i in range(n_per_dim):
                    for j in range(n_per_dim):
                        for k_idx in range(n_per_dim):
                            w = weights_1d[i] * weights_1d[j] * weights_1d[k_idx]
                            v = (nodes_1d[i] ** pi *
                                 nodes_1d[j] ** pj *
                                 nodes_1d[k_idx] ** pk)
                            approx += w * v

                if abs(exact) < 1e-30:
                    rel_err = abs(approx)
                else:
                    rel_err = abs(approx - exact) / abs(exact)
                errors.append({
                    'degree': total_deg,
                    'powers': (pi, pj, pk),
                    'exact': exact,
                    'approx': approx,
                    'rel_error': rel_err,
                })

    return errors


def compute_expectation_via_quadrature(func, n_modes, quad_order):
    """通过张量积 Gauss-Hermite 求积计算随机期望.

    E[g(ξ)] = ∫ g(ξ) · φ(ξ) dξ ≈ Σ w_k · g(ξ_k)

    对于高维 (K>3), 使用稀疏网格或 MC 替代.

    参数
    ----
    func : callable(ξ) -> float
        被积函数, ξ ∈ ℝ^K
    n_modes : int  K
    quad_order : int

    返回
    ----
    expectation : float
    n_evaluations : int
    """
    nodes_1d, weights_1d = gauss_hermite_nodes_weights(quad_order)

    if n_modes == 1:
        # 1D 求积
        expectation = sum(
            weights_1d[k] * func(np.array([nodes_1d[k]]))
            for k in range(quad_order)
        )
        return expectation, quad_order

    elif n_modes == 2:
        # 2D 张量积
        expectation = 0.0
        n_eval = 0
        for i in range(quad_order):
            for j in range(quad_order):
                xi = np.array([nodes_1d[i], nodes_1d[j]])
                w = weights_1d[i] * weights_1d[j]
                expectation += w * func(xi)
                n_eval += 1
        return expectation, n_eval

    else:
        # 高维: 使用 Monte Carlo 近似
        rng = np.random.default_rng(42)
        n_mc = min(quad_order ** n_modes, 10000)
        samples = rng.standard_normal((n_mc, n_modes))
        values = np.array([func(samples[m]) for m in range(n_mc)])
        return np.mean(values), n_mc
