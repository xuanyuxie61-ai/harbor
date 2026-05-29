# -*- coding: utf-8 -*-
"""
quadrature_integrals.py
高维数值积分与精确性验证

核心物理：
  在分数量子霍尔效应中，多体期望值涉及高维积分。
  例如Laughlin波函数的归一化常数、能量期望值等。

  对于d维积分，Gauss-Legendre求积公式为：
      ∫_{-1}^1 f(x) dx ≈ Σ_{i=1}^n w_i f(x_i)

  其中 x_i 为Legendre多项式 P_n(x) 的零点，权重为：
      w_i = 2 / [(1 - x_i²) (P_n'(x_i))²]

  多维奇异积分（库仑势）需要特殊处理：
      ∫ d²r V_C(r) = ∫_0^∞ r dr ∫_0^{2π} dθ · (e²/4πεr)

  对于楔形体上的积分（用于准粒子激发的角向分析），
  采用区域分解：
      ∫_Wedge f(x,y,z) dV = ∫_{-1}^1 dz ∫_0^1 dy ∫_0^{1-y} f(x,y,z) dx

  多项式精确性检验：对单项式 x^p y^q z^r，
  楔形体上的精确积分为：
      I(p,q,r) = [∏_{i=1}^q i / ∏_{j=p+1}^{p+q+2} j] · 2/(r+1)  (r even)

本模块融合原项目：
  - 1406_wedge_exactness（楔形体积分精确性检验）
"""
import numpy as np
from scipy.special import factorial

# ============================================================================
# 1. 组合枚举（comp_next，融合原项目 1406_wedge_exactness）
# ============================================================================

def comp_next(n, k, a, more, h, t):
    """
    生成整数n分成k个非负整数部分的所有组合。

    例如 n=3, k=2 时生成：
        (3,0), (2,1), (1,2), (0,3)

    参数:
        n    : int, 被分割的整数
        k    : int, 部分数
        a    : list, 当前组合（首次调用设为空列表或None）
        more : bool, 是否还有更多组合
        h, t : int, 内部状态参数

    返回:
        a, more, h, t
    """
    if not more:
        t = n
        h = 0
        a = [0] * k
        a[0] = n
    else:
        if t > 1:
            h = 0
        h += 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] += 1

    more = (a[k - 1] != n)
    return a, more, h, t


# ============================================================================
# 2. 单项式求值
# ============================================================================

def monomial_value(m_dim, n_points, exponents, x):
    """
    计算单项式的值：
        v_i = ∏_{j=1}^m x_j^{e_j}

    参数:
        m_dim    : int, 空间维数
        n_points : int, 求值点数
        exponents: list, 长度m的指数列表
        x        : ndarray, shape (m, n_points)

    返回:
        v        : ndarray, shape (n_points,)
    """
    x = np.asarray(x, dtype=float)
    v = np.ones(n_points, dtype=float)
    for i in range(m_dim):
        if exponents[i] != 0:
            v *= x[i, :] ** exponents[i]
    return v


# ============================================================================
# 3. 楔形体积分
# ============================================================================

def wedge01_volume():
    """
    单位楔形体体积：
        V = ∫_0^1 dx ∫_0^{1-x} dy ∫_{-1}^1 dz = 1
    """
    return 1.0


def wedge01_integral(exponents):
    """
    单位楔形体上的单项式精确积分：
        I(e1,e2,e3) = ∫_Wedge x^{e1} y^{e2} z^{e3} dV

    解析公式：
        I = [∏_{i=1}^{e2} i / ∏_{j=e1+1}^{e1+e2+2} j]
            × (2/(e3+1)  if e3 even else 0)
    """
    e = list(exponents)
    e1, e2, e3 = e[0], e[1], e[2]

    value = 1.0
    k = e1
    for i in range(1, e2 + 1):
        k += 1
        value *= i / k
    k += 1
    value /= k
    k += 1
    value /= k

    if e3 == -1:
        raise ValueError("e3 = -1 非法")
    elif e3 % 2 == 1:
        value = 0.0
    else:
        value *= 2.0 / (e3 + 1)

    return value


# ============================================================================
# 4. 高维Gauss-Legendre积分
# ============================================================================

def gauss_legendre_1d(n, a=-1.0, b=1.0):
    """
    一维Gauss-Legendre求积节点和权重。

    参数:
        n : int, 节点数
        a, b : float, 积分区间 [a, b]

    返回:
        x : ndarray, 节点
        w : ndarray, 权重
    """
    from numpy.polynomial.legendre import leggauss
    xi, wi = leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    w = 0.5 * (b - a) * wi
    return x, w


def multidimensional_gauss_legendre(f, dims, n_per_dim, domain):
    """
    多维Gauss-Legendre积分。

    参数:
        f          : callable, f(x1, x2, ..., xd) → float
        dims       : int, 维数
        n_per_dim  : int, 每维节点数
        domain     : list of tuples, [(a1,b1), (a2,b2), ...]

    返回:
        integral   : float
    """
    if len(domain) != dims:
        raise ValueError("domain 长度必须等于 dims")

    # 生成张量积网格
    grids = []
    weights = []
    for d in range(dims):
        a, b = domain[d]
        x_d, w_d = gauss_legendre_1d(n_per_dim, a, b)
        grids.append(x_d)
        weights.append(w_d)

    # 计算张量积
    total = 0.0
    if dims == 1:
        for i in range(n_per_dim):
            total += weights[0][i] * f(grids[0][i])
    elif dims == 2:
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                total += weights[0][i] * weights[1][j] * f(grids[0][i], grids[1][j])
    elif dims == 3:
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                for k in range(n_per_dim):
                    total += weights[0][i] * weights[1][j] * weights[2][k] * \
                             f(grids[0][i], grids[1][j], grids[2][k])
    else:
        raise NotImplementedError("仅支持1-3维积分")

    return total


# ============================================================================
# 5. 楔形体积分精确性检验
# ============================================================================

def wedge_exactness_test(quad_points, quad_weights, degree_max=5):
    """
    检验给定求积规则在楔形体上的多项式精确性。

    参数:
        quad_points  : ndarray, shape (3, n), 求积节点
        quad_weights : ndarray, shape (n,), 求积权重
        degree_max   : int, 检验的最大总次数

    返回:
        results      : list of (degree, exponents, quad_val, exact_val, error)
    """
    n = quad_points.shape[1]
    results = []

    for degree in range(degree_max + 1):
        a = []
        more = False
        h = 0
        t = 0
        while True:
            a, more, h, t = comp_next(degree, 3, a, more, h, t)
            exponents = a.copy()

            v = monomial_value(3, n, exponents, quad_points)
            quad_val = wedge01_volume() * np.dot(quad_weights, v)
            exact_val = wedge01_integral(exponents)
            error = abs(quad_val - exact_val)

            results.append((degree, exponents, quad_val, exact_val, error))

            if not more:
                break

    return results


# ============================================================================
# 6. 量子霍尔系统中的特殊积分
# ============================================================================

def integrate_coulomb_2d_gauss(n_radial, n_angular, epsilon_r=12.0, R_max=5.0):
    """
    使用Gauss求积计算二维库仑势的积分：
        I = ∫_0^{R_max} dr ∫_0^{2π} dθ · r · V_C(r)
          = ∫_0^{R_max} dr ∫_0^{2π} dθ · r · (e²/4πεr)
          = e²/(2ε) · R_max

    参数:
        n_radial   : int, 径向节点数
        n_angular  : int, 角向节点数
        epsilon_r  : float, 相对介电常数
        R_max      : float, 截断半径

    返回:
        integral   : float
        exact      : float
    """
    # 径向：映射 [0, R_max]
    r_nodes, r_weights = gauss_legendre_1d(n_radial, 0.0, R_max)

    # 角向：均匀分布（矩形法则等权重）
    dtheta = 2.0 * np.pi / n_angular

    integral = 0.0
    for i in range(n_radial):
        r = r_nodes[i]
        wr = r_weights[i]
        for j in range(n_angular):
            theta = j * dtheta
            # 库仑势（含短程截断）
            a_cutoff = 0.01
            r_safe = np.sqrt(r ** 2 + a_cutoff ** 2)
            V_c = 1.0 / (epsilon_r * r_safe)
            # Jacobian = r
            integral += wr * dtheta * r * V_c

    exact = 1.0 / epsilon_r * R_max  # 解析结果：e²R_max/(2ε) 在自然单位下
    return integral, exact


# ============================================================================
# 7. 测试接口
# ============================================================================
def test_quadrature_integrals():
    """测试数值积分模块。"""
    print("=" * 60)
    print("[quadrature_integrals.py] 数值积分测试")
    print("=" * 60)

    # 测试组合枚举
    print("\n1. 组合枚举测试 (n=3, k=2):")
    a = []
    more = False
    h, t = 0, 0
    count = 0
    while True:
        a, more, h, t = comp_next(3, 2, a, more, h, t)
        count += 1
        print(f"   组合 {count}: {a}")
        if not more:
            break

    # 测试楔形体精确积分
    print("\n2. 楔形体精确积分测试:")
    for e in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 2]]:
        val = wedge01_integral(e)
        print(f"   I{e} = {val:.6f}")

    # 测试多维Gauss积分
    print("\n3. 多维Gauss积分测试:")
    # 2D: ∫_0^1 ∫_0^1 x²y² dxdy = 1/9
    def f2(x, y):
        return x ** 2 * y ** 2
    val = multidimensional_gauss_legendre(f2, 2, 4, [(0.0, 1.0), (0.0, 1.0)])
    print(f"   ∫_0^1∫_0^1 x²y² dxdy = {val:.8f} (精确=1/9≈{1/9:.8f})")

    # 3D: ∫_{-1}^1 ∫_{-1}^1 ∫_{-1}^1 (x²+y²+z²) dxdydz = 8
    def f3(x, y, z):
        return x ** 2 + y ** 2 + z ** 2
    val = multidimensional_gauss_legendre(f3, 3, 4, [(-1.0, 1.0)] * 3)
    print(f"   ∫_{-1}^1³ (x²+y²+z²) dV = {val:.6f} (精确=8)")

    # 测试楔形体精确性检验
    print("\n4. 楔形体求积精确性检验:")
    # 构造简单的梯形规则节点（仅用于测试程序流程）
    n = 8
    x_pts = np.random.rand(3, n)
    # 将随机点映射到楔形体：x∈[0,1], y∈[0,1-x], z∈[-1,1]
    x_pts[1, :] *= (1.0 - x_pts[0, :])
    x_pts[2, :] = 2.0 * x_pts[2, :] - 1.0
    w = np.ones(n) / n
    results = wedge_exactness_test(x_pts, w, degree_max=2)
    max_err = max([r[4] for r in results])
    print(f"   随机求积规则最大误差 (degree≤2): {max_err:.4f}")

    # 测试2D库仑积分
    print("\n5. 二维库仑势积分测试:")
    val, exact = integrate_coulomb_2d_gauss(10, 24, epsilon_r=12.0, R_max=2.0)
    print(f"   数值积分 = {val:.6f}")
    print(f"   精确值   = {exact:.6f}")
    print(f"   相对误差 = {abs(val - exact) / abs(exact):.2e}")

    print("\n[quadrature_integrals.py] 测试完成。\n")


if __name__ == "__main__":
    test_quadrature_integrals()
