"""
quadrature_integration.py
=========================

Gauss-Laguerre 与 Gauss-Hermite 求积模块。

融合种子项目：
    467_gen_laguerre_rule：广义 Gauss-Laguerre 求积规则生成
    519_hermite_exactness：Gauss-Hermite 求积精确性检验

科学应用：
    1. Gauss-Laguerre 求积用于计算 Boltzmann 权重积分：
       ∫_0^∞ f(E) exp(-β E) dE ≈ Σ w_i f(E_i)
       其中 E_i, w_i 为 Laguerre 节点和权重

    2. Gauss-Hermite 求积用于计算活化能分布积分：
       ∫_{-∞}^{+∞} g(E_a) exp(-E_a^2) dE_a ≈ Σ w_i g(E_a,i)

数学基础：
    Gauss-Laguerre：权函数 w(x) = x^α exp(-x), x ∈ [0, ∞)
    Gauss-Hermite：权函数 w(x) = exp(-x^2), x ∈ (-∞, +∞)

作者: DA-Synthesis
"""

import math
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


# ============================================================
#  Gauss-Laguerre 求积（参考 467_gen_laguerre_rule）
# ============================================================

def laguerre_nodes_weights(order, alpha=0.0):
    """
    计算广义 Gauss-Laguerre 求积节点和权重。

    使用 Golub-Welsch 算法（三对角矩阵本征值问题）。

    广义 Laguerre 多项式三项递推：
        (n+1) L_{n+1}^(α)(x) = (2n + α + 1 - x) L_n^(α)(x)
                                - (n + α) L_{n-1}^(α)(x)

    Jacobi 矩阵元素：
        a_n = 2n + α + 1
        b_n = sqrt(n(n + α))

    Parameters
    ----------
    order : int
        求积阶数（节点数）。
    alpha : float
        权函数指数 α > -1。

    Returns
    -------
    nodes : list[float]
        求积节点 x_i。
    weights : list[float]
        求积权重 w_i。
    """
    if order < 1:
        return [], []
    if order == 1:
        return [alpha + 1.0], [math.gamma(alpha + 1.0)]

    # 构造 Jacobi 三对角矩阵
    a_diag = [0.0] * order
    b_diag = [0.0] * max(order - 1, 0)

    for n in range(order):
        a_diag[n] = 2.0 * n + alpha + 1.0

    for n in range(1, order):
        b_diag[n - 1] = math.sqrt(n * (n + alpha))

    # QR 算法求本征值和本征向量（简化版）
    nodes, vecs = _symmetric_tridiagonal_eig(a_diag, b_diag)

    # 权重 = Gamma(alpha+1) * v_0^2
    gamma_alpha = math.gamma(alpha + 1.0) if alpha > -1.0 else 1.0
    weights = [gamma_alpha * vecs[i][0] ** 2 for i in range(order)]

    # 按节点排序
    combined = sorted(zip(nodes, weights), key=lambda x: x[0])
    nodes = [c[0] for c in combined]
    weights = [c[1] for c in combined]

    return nodes, weights


def _symmetric_tridiagonal_eig(a_diag, b_diag, max_iter=100):
    """
    对称三对角矩阵的 QR 本征值算法（简化实现）。

    Parameters
    ----------
    a_diag : list[float]
        主对角线。
    b_diag : list[float]
        下次对角线。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    eigenvalues : list[float]
        本征值。
    eigenvectors : list[list[float]]
        本征向量（列存储）。
    """
    n = len(a_diag)
    if n == 0:
        return [], []

    # 拷贝
    d = list(a_diag) + [0.0]
    e = list(b_diag) + [0.0, 0.0]

    # 单位矩阵作为初始本征向量
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for iteration in range(max_iter * n):
        # 查找小的非对角元
        for m in range(n - 1, -1, -1):
            if abs(e[m]) < 1.0e-14 * (abs(d[m]) + abs(d[m + 1]) + 1.0e-30):
                e[m] = 0.0
                break

        if m == n - 1:
            # 最后一个已收敛
            continue

        # QR 步
        g = (d[m + 1] - d[m]) / (2.0 * e[m]) if abs(e[m]) > 1.0e-30 else 0.0
        r = math.sqrt(g * g + 1.0)
        shift = d[m] - e[m] / (g + r if g >= 0 else g - r)

        s = 1.0
        c = 1.0
        p = 0.0

        for i in range(m, -1, -1):
            f = s * e[i]
            b_val = c * e[i]

            if abs(f) >= abs(shift):
                c_val = shift / f
                r_val = math.sqrt(c_val * c_val + 1.0)
                e[i + 1] = f * r_val
                s = 1.0 / r_val
                c_val_new = c_val * s
            else:
                s = f / shift if abs(shift) > 1.0e-30 else 0.0
                r_val = math.sqrt(s * s + 1.0)
                e[i + 1] = shift * r_val
                c_val_new = 1.0 / r_val
                s = s * c_val_new  # 这里简化

            # 实际 QR 迭代太复杂，简化为直接对角化
            break

        # 简化：使用幂迭代近似
        break

    # 对于小规模问题，使用简化的本征值求解
    # 回退到直接法
    eigenvalues, eigenvectors = _direct_tridiag_eig(d[:n], b_diag)
    return eigenvalues, eigenvectors


def _direct_tridiag_eig(diag, off_diag):
    """
    直接法求解小规模三对角矩阵本征问题。

    Parameters
    ----------
    diag : list[float]
        主对角线。
    off_diag : list[float]
        下次对角线。

    Returns
    -------
    eigenvalues, eigenvectors
    """
    n = len(diag)
    if n == 0:
        return [], []
    if n == 1:
        return [diag[0]], [[1.0]]

    # 构造完整矩阵
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = diag[i]
        if i < n - 1:
            M[i][i + 1] = off_diag[i]
            M[i + 1][i] = off_diag[i]

    # Jacobi 本征值算法
    return _jacobi_eigen(M, n)


def _jacobi_eigen(M, n, max_iter=200, tol=1.0e-12):
    """
    Jacobi 本征值算法求对称矩阵的全部本征对。

    Parameters
    ----------
    M : list[list[float]]
        对称矩阵。
    n : int
        矩阵大小。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容限。

    Returns
    -------
    eigenvalues : list[float]
        本征值（降序）。
    eigenvectors : list[list[float]]
        本征向量列表。
    """
    # 拷贝矩阵
    A = [row[:] for row in M]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for iteration in range(max_iter):
        # 找最大非对角元
        max_off = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > max_off:
                    max_off = abs(A[i][j])
                    p, q = i, j

        if max_off < tol:
            break

        # 计算旋转角
        if abs(A[p][p] - A[q][q]) < 1.0e-30:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * A[p][q], A[p][p] - A[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        # 旋转
        A_new = [row[:] for row in A]
        for i in range(n):
            A_new[i][p] = c * A[i][p] + s * A[i][q]
            A_new[i][q] = -s * A[i][p] + c * A[i][q]
        A = A_new
        A_new2 = [row[:] for row in A]
        for j in range(n):
            A_new2[p][j] = c * A[p][j] + s * A[q][j]
            A_new2[q][j] = -s * A[p][j] + c * A[q][j]
        A = A_new2

        # 更新本征向量
        V_new = [row[:] for row in V]
        for i in range(n):
            V_new[i][p] = c * V[i][p] + s * V[i][q]
            V_new[i][q] = -s * V[i][p] + c * V[i][q]
        V = V_new

    eigenvalues = [A[i][i] for i in range(n)]
    eigenvectors = [[V[i][j] for i in range(n)] for j in range(n)]

    # 按本征值降序排列
    order = sorted(range(n), key=lambda k: -eigenvalues[k])
    eigenvalues = [eigenvalues[k] for k in order]
    eigenvectors = [eigenvectors[k] for k in order]

    return eigenvalues, eigenvectors


def gauss_laguerre_integrate(func, order, alpha=0.0, a=0.0, b_scale=1.0):
    """
    使用 Gauss-Laguerre 求积计算积分。

    积分形式：
        ∫_a^∞ |x-a|^α exp(-b(x-a)) f(x) dx

    Parameters
    ----------
    func : callable
        被积函数 f(x)。
    order : int
        求积阶数。
    alpha : float
        权函数指数。
    a : float
        积分下限。
    b_scale : float
        指数衰减因子。

    Returns
    -------
    float
        积分近似值。
    """
    nodes, weights = laguerre_nodes_weights(order, alpha)
    if not nodes:
        return 0.0

    result = 0.0
    for i in range(len(nodes)):
        x_i = a + nodes[i] / b_scale if abs(b_scale) > 1.0e-30 else a + nodes[i]
        w_i = weights[i] / (b_scale ** (alpha + 1.0)) if abs(b_scale) > 1.0e-30 else weights[i]
        result += w_i * func(x_i)

    return result


# ============================================================
#  Gauss-Hermite 求积（参考 519_hermite_exactness）
# ============================================================

def hermite_nodes_weights(order):
    """
    计算 Gauss-Hermite 求积节点和权重。

    Hermite 多项式三项递推（物理学家约定）：
        H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)

    Parameters
    ----------
    order : int
        求积阶数。

    Returns
    -------
    nodes : list[float]
        节点。
    weights : list[float]
        权重。
    """
    if order < 1:
        return [], []

    # Jacobi 矩阵：a_n = 0, b_n = sqrt(n/2)
    diag = [0.0] * order
    off_diag = [math.sqrt(i / 2.0) for i in range(1, order)]

    eigenvalues, eigenvectors = _direct_tridiag_eig(diag, off_diag)

    # 权重 = sqrt(pi) * v_0^2
    weights = [math.sqrt(math.pi) * vec[0] ** 2 for vec in eigenvectors]

    combined = sorted(zip(eigenvalues, weights), key=lambda x: x[0])
    nodes = [c[0] for c in combined]
    weights = [c[1] for c in combined]

    return nodes, weights


def hermite_exactness_test(order, degree_max):
    """
    检验 Gauss-Hermite 求积的多项式精确性（参考 519）。

    N 阶 Gauss-Hermite 应精确积分次数 ≤ 2N-1 的多项式。

    Parameters
    ----------
    order : int
        求积阶数。
    degree_max : int
        检验的最大多项式次数。

    Returns
    -------
    results : list[dict]
        各次数的精确性检验结果。
    """
    nodes, weights = hermite_nodes_weights(order)
    results = []

    for n in range(degree_max + 1):
        # 精确值：∫ x^n exp(-x^2) dx
        if n % 2 == 1:
            exact = 0.0
        else:
            exact = math.gamma((n + 1.0) / 2.0)

        # 数值积分
        approx = sum(w * xi ** n for xi, w in zip(nodes, weights))

        error = abs(approx - exact)
        results.append({
            "degree": n,
            "exact": exact,
            "approx": approx,
            "error": error,
            "exact_within_tolerance": error < 1.0e-8,
        })

    return results


def gauss_hermite_integrate(func, order):
    """
    使用 Gauss-Hermite 求积计算积分。

    积分形式：
        ∫_{-∞}^{+∞} f(x) exp(-x^2) dx ≈ Σ w_i f(x_i)

    Parameters
    ----------
    func : callable
        被积函数 f(x)。
    order : int
        求积阶数。

    Returns
    -------
    float
        积分近似值。
    """
    nodes, weights = hermite_nodes_weights(order)
    return sum(w * func(xi) for xi, w in zip(nodes, weights))


# ============================================================
#  SEI 应用：Boltzmann 权重活化能积分
# ============================================================

def boltzmann_averaged_rate(k0, activation_energy_mean, activation_energy_std,
                              temperature):
    """
    计算 Boltzmann 平均的反应速率（考虑活化能分布）。

    数学形式：
        <k> = ∫ k0 exp(-E_a/(k_B T)) p(E_a) dE_a
    其中 p(E_a) 为高斯分布。

    Parameters
    ----------
    k0 : float
        指前因子。
    activation_energy_mean : float
        平均活化能 [J/mol]。
    activation_energy_std : float
        活化能标准差 [J/mol]。
    temperature : float
        温度 [K]。

    Returns
    -------
    float
        平均反应速率。
    """
    kb = P.R_GAS  # J/(mol K)
    beta = 1.0 / (kb * temperature)

    def integrand(ea):
        return math.exp(-beta * ea)

    # 使用 Hermite 求积（将高斯积分变换到标准形式）
    sigma = activation_energy_std
    mu = activation_energy_mean

    def transformed_integrand(t):
        ea = mu + sigma * t
        return math.exp(-beta * ea)

    order = P.GH_ORDER
    result = gauss_hermite_integrate(transformed_integrand, order)
    return k0 * result / math.sqrt(math.pi)


# ============================================================
#  综合演示
# ============================================================

def run_quadrature_demo():
    """
    运行求积演示。

    Returns
    -------
    dict
        求积分析结果。
    """
    # Laguerre 节点权重
    gl_nodes, gl_weights = laguerre_nodes_weights(P.GL_ORDER, P.GL_ALPHA_PARAM)

    # Hermite 精确性检验
    hermite_test = hermite_exactness_test(P.GH_ORDER, 2 * P.GH_ORDER)

    # Boltzmann 平均速率
    ea_mean = 50000.0  # J/mol
    ea_std = 5000.0    # J/mol
    avg_rate = boltzmann_averaged_rate(
        P.K0_SEI, ea_mean, ea_std, P.T_OPER
    )

    return {
        "laguerre_nodes": gl_nodes,
        "laguerre_weights": gl_weights,
        "hermite_exactness": hermite_test,
        "boltzmann_avg_rate": avg_rate,
    }


if __name__ == "__main__":
    result = run_quadrature_demo()
    print(f"[quadrature] Gauss-Laguerre 节点 (阶 {P.GL_ORDER}):")
    for i, (x, w) in enumerate(
            zip(result['laguerre_nodes'], result['laguerre_weights'])):
        print(f"  x_{i} = {x:.6f}, w_{i} = {w:.6e}")

    print(f"\n[quadrature] Gauss-Hermite 精确性检验 (阶 {P.GH_ORDER}):")
    for r in result['hermite_exactness']:
        status = "✓" if r['exact_within_tolerance'] else "✗"
        print(f"  deg {r['degree']:2d}: exact={r['exact']:.6e}, "
              f"approx={r['approx']:.6e}, err={r['error']:.2e} {status}")

    print(f"\n[quadrature] Boltzmann 平均速率 = {result['boltzmann_avg_rate']:.6e}")
