"""
thermodynamic_quadrature.py
热力学积分与几何矩计算模块

核心功能：
- 广义 Gauss-Laguerre 正交规则生成
- Jacobi 矩阵构造与 IMTQLX 特征值算法
- 多边形和正六边形区域上的矩积分
- 热力学积分（自由能计算）

科学背景：
酶催化反应的自由能面计算需要高维积分：
    ΔG(ξ) = -k_B T ln ∫ δ(ξ(r) - ξ') exp(-V(r)/(k_B T)) dr

热力学积分方法：
    ΔG = ∫_0^1 ⟨∂V/∂λ⟩_λ dλ

其中 λ 为反应进度参数。对于各 λ 点，需要计算系综平均 ⟨·⟩。

广义 Gauss-Laguerre 正交：
    ∫_a^∞ (x-a)^α exp(-b(x-a)) f(x) dx ≈ Σ_{i=1}^n w_i f(x_i)

节点 x_i 和权重 w_i 由 Jacobi 矩阵的特征值和特征向量确定：
    J = [ α_0   β_1    0     ...   ]
        [ β_1   α_1   β_2    ...   ]
        [  0    β_2   α_2    ...   ]
        [ ...   ...   ...    ...   ]

其中：
    α_i = 2i + 1 + α
    β_i = √(i(i + α))

特征值 → 节点 x_i
特征向量首分量平方 → 权重 w_i

---
多边形矩积分（Steger 方法）：
    ν_{pq} = ∮_P x^p y^q dx dy
           = Σ_{边} (x_j y_i - x_i y_j) * S_{pq} / [(p+q+2)(p+q+1)C(p+q,p)]

其中 S_{pq} 为边上的求和项。

在过渡态理论中，矩积分用于计算：
    - 反应盆地的惯性矩（决定转动配分函数）
    - 构型空间体积（决定平动配分函数）
    - 有效质量张量
"""

import numpy as np
from math import comb


class IMTQLX:
    """
    隐式 QL 算法（对角化对称三对角矩阵）

    输入：
        d: 对角元
        e: 次对角元（e[0] 未使用）
        z: 初始向量
    输出：
        d: 特征值（升序排列）
        z: Q^T z
    """

    @staticmethod
    def diagonalize(d, e, z, max_iter=30):
        d = np.asarray(d, dtype=float).copy()
        e = np.asarray(e, dtype=float).copy()
        z = np.asarray(z, dtype=float).copy()
        n = len(d)

        if n == 1:
            return d, z

        e[n - 1] = 0.0
        prec = np.finfo(float).eps

        for l in range(n):
            j = 0
            while True:
                for m in range(l, n):
                    if m == n - 1:
                        break
                    if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                        break

                p = d[l]
                if m == l:
                    break

                if j == max_iter:
                    raise RuntimeError("IMTQLX: 迭代次数超过上限")
                j += 1

                g = (d[l + 1] - p) / (2.0 * e[l])
                r = np.sqrt(g * g + 1.0)
                g = d[m] - p + e[l] / (g + np.sign(g) * abs(r))
                s = 1.0
                c = 1.0
                p = 0.0
                mml = m - l

                for ii in range(1, mml + 1):
                    i = m - ii
                    f = s * e[i]
                    b = c * e[i]

                    if abs(f) >= abs(g):
                        c = g / f
                        r = np.sqrt(c * c + 1.0)
                        e[i + 1] = f * r
                        s = 1.0 / r
                        c *= s
                    else:
                        s = f / g
                        r = np.sqrt(s * s + 1.0)
                        e[i + 1] = g * r
                        c = 1.0 / r
                        s *= c

                    g = d[i + 1] - p
                    r = (d[i] - g) * s + 2.0 * c * b
                    p = s * r
                    d[i + 1] = g + p
                    g = c * r - b
                    f = z[i + 1]
                    z[i + 1] = s * z[i] + c * f
                    z[i] = c * z[i] - s * f

                d[l] -= p
                e[l] = g
                e[m] = 0.0

        # 按特征值排序
        for ii in range(1, n):
            i = ii - 1
            k = i
            p = d[i]
            for j in range(ii, n):
                if d[j] < p:
                    k = j
                    p = d[j]
            if k != i:
                d[k] = d[i]
                d[i] = p
                p = z[i]
                z[i] = z[k]
                z[k] = p

        return d, z


class GaussLaguerreQuadrature:
    """
    广义 Gauss-Laguerre 正交规则生成器

    积分：
        ∫_a^∞ (x-a)^α exp(-b(x-a)) f(x) dx
    """

    def __init__(self, order, alpha_param, a=0.0, b=1.0):
        """
        参数：
            order: 正交点数量
            alpha_param: 幂次参数 α > -1
            a: 左端点
            b: 指数缩放因子
        """
        if alpha_param <= -1.0:
            raise ValueError("alpha 必须 > -1")
        if b <= 0.0:
            raise ValueError("b 必须 > 0")

        self.order = order
        self.alpha = alpha_param
        self.a = a
        self.b = b
        self.x = None
        self.w = None
        self._compute_rule()

    def _compute_rule(self):
        """
        构造 Jacobi 矩阵并计算节点和权重

        对于 Laguerre 型：
            α_i = 2i + 1 + α
            β_i = i(i + α)
        """
        m = self.order
        aj = np.zeros(m, dtype=float)
        bj = np.zeros(m, dtype=float)

        # 广义 Laguerre 的 Jacobi 矩阵元素
        for i in range(m):
            aj[i] = 2.0 * i + 1.0 + self.alpha
            bj[i] = (i + 1) * (i + 1 + self.alpha)

        # 零阶矩
        from math import gamma
        zemu = gamma(self.alpha + 1.0)

        # 使用 IMTQLX 对角化
        z = np.zeros(m, dtype=float)
        z[0] = np.sqrt(zemu)
        d, w = IMTQLX.diagonalize(aj, np.sqrt(bj), z)

        # 权重为特征向量首分量平方
        w = w ** 2

        # 缩放：Laguerre 区间 [a, ∞)
        # 标准 Laguerre 节点在 [0,∞)，权重满足 ∫ x^α exp(-x) dx = Γ(α+1)
        # 经缩放 x' = a + x/b 后：
        #   ∫_a^∞ (x'-a)^α exp(-b(x'-a)) f(x') dx'
        #   = (1/b^{α+1}) ∫_0^∞ x^α exp(-x) f(a + x/b) dx
        self.x = self.a + d / self.b
        self.w = w / (self.b ** (self.alpha + 1.0))

    def integrate(self, func):
        """
        积分函数 func
        """
        return np.sum(self.w * func(self.x))


class PolygonMoments:
    """
    多边形矩积分计算（Steger 算法）
    """

    @staticmethod
    def r8_mop(i):
        """返回 (-1)^i"""
        return 1.0 if i % 2 == 0 else -1.0

    @classmethod
    def moment_unnormalized(cls, n, x, y, p, q):
        """
        计算非归一化矩：
            ν_{pq} = ∮_P x^p y^q dx dy
        """
        nu_pq = 0.0
        xj = x[n - 1]
        yj = y[n - 1]

        for i in range(n):
            xi = x[i]
            yi = y[i]
            s_pq = 0.0
            for k in range(p + 1):
                for l in range(q + 1):
                    s_pq += (comb(k + l, l) * comb(p + q - k - l, q - l) *
                             xi ** k * xj ** (p - k) * yi ** l * yj ** (q - l))
            nu_pq += (xj * yi - xi * yj) * s_pq
            xj = xi
            yj = yi

        denom = (p + q + 2) * (p + q + 1) * comb(p + q, p)
        return nu_pq / denom

    @classmethod
    def moment_normalized(cls, n, x, y, p, q):
        """
        归一化矩：
            α_{pq} = ν_{pq} / ν_{00}
        """
        nu_pq = cls.moment_unnormalized(n, x, y, p, q)
        nu_00 = cls.moment_unnormalized(n, x, y, 0, 0)
        if abs(nu_00) < 1e-15:
            raise ValueError("多边形面积为零")
        return nu_pq / nu_00

    @classmethod
    def moment_central(cls, n, x, y, p, q):
        """
        中心矩：
            μ_{pq} = (1/A) ∮_P (x - x̄)^p (y - ȳ)^q dx dy
        """
        alpha_10 = cls.moment_normalized(n, x, y, 1, 0)
        alpha_01 = cls.moment_normalized(n, x, y, 0, 1)

        mu_pq = 0.0
        for i in range(p + 1):
            for j in range(q + 1):
                alpha_ij = cls.moment_normalized(n, x, y, i, j)
                mu_pq += (cls.r8_mop(p + q - i - j) * comb(p, i) * comb(q, j) *
                          alpha_10 ** (p - i) * alpha_01 ** (q - j) * alpha_ij)
        return mu_pq


class HexagonMoments:
    """
    单位正六边形矩积分
    顶点：[(1,0), (1/2, √3/2), (-1/2, √3/2), (-1,0), (-1/2,-√3/2), (1/2,-√3/2)]
    """

    def __init__(self):
        a = np.sqrt(3.0) / 2.0
        self.n = 6
        self.x = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5], dtype=float)
        self.y = np.array([0.0, a, a, 0.0, -a, -a], dtype=float)

    def integral_monomial(self, p, q):
        """
        计算 ∫_{hexagon} x^p y^q dx dy
        若 p 或 q 为奇数，则积分为零（对称性）
        """
        if p % 2 == 1 or q % 2 == 1:
            return 0.0
        return PolygonMoments.moment_unnormalized(self.n, self.x, self.y, p, q)


class ThermodynamicIntegration:
    """
    热力学积分模块

    用于计算酶催化反应的自由能变化：
        ΔG = ∫_0^1 ⟨∂H/∂λ⟩_λ dλ

    其中 λ 为反应坐标耦合参数，H 为哈密顿量。
    """

    def __init__(self, n_lambda=20, temperature=300.0):
        self.n_lambda = n_lambda
        self.T = temperature
        self.kB = 0.0019872041  # kcal/(mol·K)
        self.beta = 1.0 / (self.kB * temperature)

    def free_energy_barrier(self, energy_profile, xi_values):
        """
        从能量剖面计算自由能势垒

        使用：
            G(ξ) = -k_B T ln P(ξ)
        其中 P(ξ) = exp(-E(ξ)/(k_B T)) / Z

        参数：
            energy_profile: 能量值数组（kcal/mol）
            xi_values: 反应坐标值
        返回：
            free_energy: 自由能剖面
            barrier: 活化自由能 ΔG‡
        """
        # TODO(Hole_1): 实现自由能势垒计算
        # 需要从能量剖面计算 Boltzmann 概率、配分函数 Z、自由能 G(ξ)
        # 并识别过渡态、反应物、产物的位置，返回活化自由能
        raise NotImplementedError("Hole_1: 请实现 free_energy_barrier 方法")

    def entropic_contribution(self, energy_profile, xi_values):
        """
        计算熵贡献：
            -TΔS = ΔG - ΔE
        """
        free_energy, dG, dG_rev = self.free_energy_barrier(energy_profile, xi_values)
        dE = np.max(energy_profile) - np.min(energy_profile)
        return -self.T * (dG - dE) / self.T  # -TΔS
