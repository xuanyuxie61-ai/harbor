# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import factorial





def comp_next(n, k, a, more, h, t):
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






def monomial_value(m_dim, n_points, exponents, x):
    x = np.asarray(x, dtype=float)
    v = np.ones(n_points, dtype=float)
    for i in range(m_dim):
        if exponents[i] != 0:
            v *= x[i, :] ** exponents[i]
    return v






def wedge01_volume():
    return 1.0


def wedge01_integral(exponents):
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






def gauss_legendre_1d(n, a=-1.0, b=1.0):
    from numpy.polynomial.legendre import leggauss
    xi, wi = leggauss(n)
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    w = 0.5 * (b - a) * wi
    return x, w


def multidimensional_gauss_legendre(f, dims, n_per_dim, domain):
    if len(domain) != dims:
        raise ValueError("domain 长度必须等于 dims")


    grids = []
    weights = []
    for d in range(dims):
        a, b = domain[d]
        x_d, w_d = gauss_legendre_1d(n_per_dim, a, b)
        grids.append(x_d)
        weights.append(w_d)


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






def wedge_exactness_test(quad_points, quad_weights, degree_max=5):
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






def integrate_coulomb_2d_gauss(n_radial, n_angular, epsilon_r=12.0, R_max=5.0):

    r_nodes, r_weights = gauss_legendre_1d(n_radial, 0.0, R_max)


    dtheta = 2.0 * np.pi / n_angular

    integral = 0.0
    for i in range(n_radial):
        r = r_nodes[i]
        wr = r_weights[i]
        for j in range(n_angular):
            theta = j * dtheta

            a_cutoff = 0.01
            r_safe = np.sqrt(r ** 2 + a_cutoff ** 2)
            V_c = 1.0 / (epsilon_r * r_safe)

            integral += wr * dtheta * r * V_c

    exact = 1.0 / epsilon_r * R_max
    return integral, exact





def test_quadrature_integrals():
    print("=" * 60)
    print("[quadrature_integrals.py] 数值积分测试")
    print("=" * 60)


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


    print("\n2. 楔形体精确积分测试:")
    for e in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 2]]:
        val = wedge01_integral(e)
        print(f"   I{e} = {val:.6f}")


    print("\n3. 多维Gauss积分测试:")

    def f2(x, y):
        return x ** 2 * y ** 2
    val = multidimensional_gauss_legendre(f2, 2, 4, [(0.0, 1.0), (0.0, 1.0)])
    print(f"   ∫_0^1∫_0^1 x²y² dxdy = {val:.8f} (精确=1/9≈{1/9:.8f})")


    def f3(x, y, z):
        return x ** 2 + y ** 2 + z ** 2
    val = multidimensional_gauss_legendre(f3, 3, 4, [(-1.0, 1.0)] * 3)
    print(f"   ∫_{-1}^1³ (x²+y²+z²) dV = {val:.6f} (精确=8)")


    print("\n4. 楔形体求积精确性检验:")

    n = 8
    x_pts = np.random.rand(3, n)

    x_pts[1, :] *= (1.0 - x_pts[0, :])
    x_pts[2, :] = 2.0 * x_pts[2, :] - 1.0
    w = np.ones(n) / n
    results = wedge_exactness_test(x_pts, w, degree_max=2)
    max_err = max([r[4] for r in results])
    print(f"   随机求积规则最大误差 (degree≤2): {max_err:.4f}")


    print("\n5. 二维库仑势积分测试:")
    val, exact = integrate_coulomb_2d_gauss(10, 24, epsilon_r=12.0, R_max=2.0)
    print(f"   数值积分 = {val:.6f}")
    print(f"   精确值   = {exact:.6f}")
    print(f"   相对误差 = {abs(val - exact) / abs(exact):.2e}")

    print("\n[quadrature_integrals.py] 测试完成。\n")


if __name__ == "__main__":
    test_quadrature_integrals()
