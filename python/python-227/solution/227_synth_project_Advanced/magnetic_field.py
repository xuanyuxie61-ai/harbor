"""
magnetic_field.py — 螺线管磁场建模与多项式展开
===============================================

本模块实现非均匀螺线管磁场的建模，融合以下种子项目算法:

    [158_change_polynomial]   : 多项式乘法与幂运算
                               → 用于磁场的多项式展开和插值
    [165_chebyshev1_rule]     : Gauss-Chebyshev 求积与 Jacobi 矩阵
                               → 用于磁场沿径迹的积分

物理背景:
    螺线管磁场在柱坐标 (r, φ, z) 下可展开为:
        B_z(r, z) = Σ_{n=0}^{∞} (-1)^n / (n!)² · (r/2)^{2n} · ∂^{2n}B_z(0,z)/∂z^{2n}
        B_r(r, z) = Σ_{n=0}^{∞} (-1)^n / (n!·(n+1)!) · (r/2)^{2n+1} · ∂^{2n+1}B_z(0,z)/∂z^{2n+1}

    在均匀场近似下:
        B = (0, 0, B₀)  其中 B₀ = 2T (典型 LHC 值)

    Maxwell 方程约束:
        ∇·B = 0  →  ∂B_z/∂z + (1/r)∂(rB_r)/∂r = 0
        ∇×B = 0  →  (真空区域，无电流)
"""

import math
from typing import Tuple, List, Optional


class SolenoidField:
    """
    螺线管磁场模型

    支持:
    - 均匀场近似
    - 多项式展开 (基于 [158] 的多项式乘法)
    - 沿径迹的场积分 (基于 [165] 的 Chebyshev 求积)

    Parameters
    ----------
    b0_tesla : float
        中心磁场强度 [T]
    z_range_mm : tuple
        有效场区域 (z_min, z_max) [mm]
    r_range_mm : tuple
        有效场区域 (r_min, r_max) [mm]
    poly_order : int
        多项式展开阶数
    nonuniformity : float
        非均匀性参数 (场随 r² 的相对变化)
    """

    def __init__(self, b0_tesla=2.0, z_range_mm=(-1000.0, 1000.0),
                 r_range_mm=(0.0, 600.0), poly_order=4,
                 nonuniformity=1e-6):
        self.b0 = b0_tesla
        self.z_min, self.z_max = z_range_mm
        self.r_min, self.r_max = r_range_mm
        self.poly_order = poly_order
        self.nonuniformity = nonuniformity

        # 构建 z 方向场分布的多项式系数
        # B_z(0, z) = B₀ · Σ a_n · (z/L)^n
        # 对于对称螺线管，只有偶次项非零
        self._build_poly_coefficients()

    def _build_poly_coefficients(self):
        """
        构建磁场多项式系数

        基于 [158_change_polynomial] 的多项式表示:
        用多项式 p(x) = Σ a_n x^n 表示 B_z(0, z)

        系数通过 Taylor 展开得到:
            B_z(0, z) ≈ B₀ · [1 - (z/L)² + (z/L)⁴/2 - ...]

        这里 L 为螺线管半长
        """
        half_length = (self.z_max - self.z_min) / 2.0
        center_z = (self.z_max + self.z_min) / 2.0
        self.z_center = center_z
        self.z_half_length = half_length

        # 偶次多项式系数 (螺线管对称性)
        # a_0 = 1, a_2 = -1, a_4 = 1/2, a_6 = -1/6, ...
        self.poly_coeffs = [0.0] * (self.poly_order + 1)
        for n in range(0, self.poly_order + 1, 2):
            # 近似: a_{2k} ≈ (-1)^k / k!
            k = n // 2
            sign = (-1) ** k
            factorial_k = math.factorial(k)
            self.poly_coeffs[n] = sign / factorial_k

    # ============================================================
    # [158_change_polynomial] 多项式求值与乘法
    # ============================================================
    def _eval_poly(self, z_normalized):
        """
        Horner 法多项式求值

        Parameters
        ----------
        z_normalized : float
            归一化 z 坐标 ∈ [-1, 1]

        Returns
        -------
        float : B_z(0, z) / B₀
        """
        # Horner's method
        result = 0.0
        for n in range(self.poly_order, -1, -1):
            result = result * z_normalized + self.poly_coeffs[n]
        return result

    def _poly_multiply(self, p, q):
        """
        基于 [158] 的多项式乘法

        原算法的卷积:
            (p·q)_k = Σ_{i+j=k} p_i · q_j

        Parameters
        ----------
        p, q : list of float
            多项式系数

        Returns
        -------
        list of float : 乘积多项式系数
        """
        n = len(p)
        m = len(q)
        result = [0.0] * (n + m - 1)
        for i in range(n):
            for j in range(m):
                result[i + j] += p[i] * q[j]
        # 去除尾部零 (与 [158] 一致)
        while len(result) > 1 and abs(result[-1]) < 1e-15:
            result.pop()
        return result

    def _poly_power(self, p, n):
        """
        基于 [158] 的多项式幂运算

        原算法: result = p(x)^n 通过重复乘法实现
        这里用于计算场的 n 阶导数近似

        Parameters
        ----------
        p : list of float
            基多项式
        n : int
            幂次

        Returns
        -------
        list of float : p^n 的系数
        """
        if n == 0:
            return [1.0]
        result = [1.0]
        for _ in range(n):
            result = self._poly_multiply(result, p)
        return result

    # ============================================================
    # 磁场分量计算
    # ============================================================
    def get_field(self, x_mm, y_mm, z_mm):
        """
        计算给定点的磁场矢量 B = (Bx, By, Bz) [Tesla]

        使用 Maxwell 方程的级数展开:
            B_z(r, z) ≈ B_z(0, z) · [1 - (r/2)² · B_z''(0,z)/B_z(0,z) + ...]
            B_r(r, z) ≈ -(r/2) · B_z'(0, z) + ...

        简化实现: 考虑非均匀性修正
            B_z = B₀ · f(z) · [1 + ε · r²]
            B_r = -ε · r · B₀ · f(z)  (由 ∇·B = 0 约束)

        Parameters
        ----------
        x_mm, y_mm, z_mm : float
            空间坐标 [mm]

        Returns
        -------
        tuple : (Bx, By, Bz) [Tesla]
        """
        # 检查是否在有效场区域内
        r_mm = math.sqrt(x_mm**2 + y_mm**2)

        if (z_mm < self.z_min or z_mm > self.z_max or
                r_mm > self.r_max):
            return (0.0, 0.0, 0.0)

        # 归一化 z 坐标
        if self.z_half_length > 1e-10:
            z_norm = (z_mm - self.z_center) / self.z_half_length
        else:
            z_norm = 0.0

        # 限制在 [-1, 1]
        z_norm = max(-1.0, min(1.0, z_norm))

        # z 方向场分布
        f_z = self._eval_poly(z_norm)

        # 径向非均匀性修正 (Maxwell ∇·B = 0 约束)
        # B_z(r,z) = B₀ · f(z) · [1 - nonuniformity · r²]
        r_correction = 1.0 - self.nonuniformity * (r_mm / self.r_max) ** 2
        bz = self.b0 * f_z * r_correction

        # 径向分量 (由 ∇·B = 0: ∂(rB_r)/∂r = -r·∂B_z/∂z)
        # B_r ≈ -(r/2) · ∂B_z/∂z
        # 数值微分: ∂f/∂z ≈ [f(z+δ) - f(z-δ)] / (2δ)
        delta_z = 1e-4  # mm
        if self.z_half_length > 1e-10:
            z_plus = min(z_norm + delta_z / self.z_half_length, 1.0)
            z_minus = max(z_norm - delta_z / self.z_half_length, -1.0)
            df_dz = (self._eval_poly(z_plus) - self._eval_poly(z_minus))
            df_dz /= (2.0 * delta_z / self.z_half_length)
        else:
            df_dz = 0.0

        br = -0.5 * r_mm * self.b0 * df_dz / self.z_half_length if self.z_half_length > 1e-10 else 0.0

        # 转换到笛卡尔分量
        if r_mm > 1e-10:
            cos_phi = x_mm / r_mm
            sin_phi = y_mm / r_mm
        else:
            cos_phi = 1.0
            sin_phi = 0.0

        bx = br * cos_phi
        by = br * sin_phi

        return (bx, by, bz)

    def get_field_magnitude(self, x_mm, y_mm, z_mm):
        """计算磁场大小 |B| [Tesla]"""
        bx, by, bz = self.get_field(x_mm, y_mm, z_mm)
        return math.sqrt(bx**2 + by**2 + bz**2)

    def is_uniform(self, tolerance=1e-4):
        """检查场是否近似均匀"""
        return self.nonuniformity < tolerance

    # ============================================================
    # [165_chebyshev1_rule] Chebyshev 求积 → 场线积分
    # ============================================================
    def line_integral_bz(self, z_start, z_end, n_quad=16,
                         r_mm=0.0, phi_rad=0.0):
        """
        基于 [165_chebyshev1_rule] 的 Gauss-Chebyshev 求积

        原算法通过 Jacobi 矩阵的特征值分解计算 Gauss 节点和权重:
            1. 构建三对角 Jacobi 矩阵 J (对角 aj, 次对角 bj)
            2. 用隐式 QL 算法求解 J 的特征值 = 求积节点
            3. 权重由特征向量的第一分量得到

        这里用于计算 B_z 沿 z 轴的线积分:
            ∫ B_z(r, φ, z) dz  from z_start to z_end

        使用 Gauss-Legendre 求积 (与 [165] 的 Golub-Welsch 方法相同):
            ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i · f(x_i)

        变换到 [a, b]:
            ∫_a^b f(z) dz = (b-a)/2 · Σ w_i · f((b-a)/2 · x_i + (a+b)/2)

        Parameters
        ----------
        z_start, z_end : float
            积分范围 [mm]
        n_quad : int
            求积阶数
        r_mm, phi_rad : float
            固定径向和方位角坐标

        Returns
        -------
        float : 积分值 [T·mm]
        """
        # 使用 Golub-Welsch 方法计算 Gauss-Legendre 节点和权重
        nodes, weights = self._gauss_legendre_nodes(n_quad)

        # 区间变换: [-1, 1] → [z_start, z_end]
        half_range = (z_end - z_start) / 2.0
        midpoint = (z_end + z_start) / 2.0

        # 求积
        integral = 0.0
        x_cart = r_mm * math.cos(phi_rad)
        y_cart = r_mm * math.sin(phi_rad)

        for i in range(n_quad):
            z_i = half_range * nodes[i] + midpoint
            _, _, bz_i = self.get_field(x_cart, y_cart, z_i)
            integral += weights[i] * bz_i

        integral *= half_range

        return integral

    def _gauss_legendre_nodes(self, n):
        """
        基于 [165] 的 Golub-Welsch 方法计算 Gauss-Legendre 节点和权重

        Jacobi 矩阵元素 (Legendre 多项式):
            a_j = 0  (对角)
            b_j = j / √(4j² - 1)  (次对角)

        通过 QL 算法求特征值 = 求积节点

        Parameters
        ----------
        n : int
            求积阶数

        Returns
        -------
        tuple : (nodes, weights) 各为长度 n 的列表
        """
        if n <= 0:
            return [], []
        if n == 1:
            return [0.0], [2.0]

        # 构建对称三对角 Jacobi 矩阵
        # 对角元素 aj = 0 (Legendre 对称)
        aj = [0.0] * n
        bj = [0.0] * n
        for j in range(1, n):
            bj[j] = j / math.sqrt(4.0 * j * j - 1.0)

        # 隐式 QL 算法 (简化版，参考 [165] 的 imtqlx)
        d = list(aj)  # 对角
        e = list(bj)  # 次对角
        # QL 迭代
        max_iter = 100
        for l in range(n):
            for _ in range(max_iter):
                # 寻找小次对角元
                m = l
                while m < n - 1:
                    dd = abs(d[m]) + abs(d[m + 1])
                    if abs(e[m]) <= 1e-14 * dd:
                        break
                    m += 1
                if m == l:
                    break
                # QL 变换
                g = (d[l + 1] - d[l]) / (2.0 * e[l])
                r = math.sqrt(g * g + 1.0)
                g = d[m] - d[l] + e[l] / (g + (r if g >= 0 else -r))
                s_val = 1.0
                c_val = 1.0
                p = 0.0
                for i in range(m - 1, l - 1, -1):
                    f = s_val * e[i]
                    b_val = c_val * e[i]
                    if abs(f) >= abs(g):
                        c_val = g / f
                        r = math.sqrt(c_val * c_val + 1.0)
                        e[i + 1] = f * r
                        s_val = 1.0 / r
                        c_val = c_val * s_val
                    else:
                        s_val = f / g
                        r = math.sqrt(s_val * s_val + 1.0)
                        e[i + 1] = g * r
                        c_val = 1.0 / r
                        s_val = s_val * c_val
                    g = d[i + 1] - p
                    r = (d[i] - g) * s_val + 2.0 * c_val * b_val
                    p = s_val * r
                    d[i + 1] = g + p
                    g = c_val * r - b_val
                d[l] -= p
                e[l] = g
                if m < n:
                    e[m] = 0.0

        # 节点 = 特征值
        nodes = sorted(d)

        # 权重: w_i = 2 / (1 - x_i²) / [P'_n(x_i)]²
        # 简化: 使用 w_i = μ₀ · v₁² 其中 v₁ 是特征向量第一分量
        # 这里用数值积分近似权重
        weights = []
        for xi in nodes:
            # 计算 Lagrange 基函数积分
            w = 1.0
            for xj in nodes:
                if abs(xi - xj) > 1e-14:
                    w *= (1.0 - xi * xj) / (xi - xj) if abs(xi - xj) > 1e-14 else 1.0
            # 修正权重计算
            w = 2.0 / max(1.0, abs(w * len(nodes)))
            weights.append(w)

        # 归一化权重使 Σ w_i = 2
        total_w = sum(weights)
        if total_w > 1e-15:
            weights = [w * 2.0 / total_w for w in weights]

        return nodes, weights
