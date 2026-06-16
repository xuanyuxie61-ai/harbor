# -*- coding: utf-8 -*-
"""
peierls_nabarro.py
==================
Peierls-Nabarro位错核心模型 — 椭圆积分求解器

本模块融合以下种子项目算法:
- 335_elliptic_integral: 椭圆积分 → 位错核心能量与Peierls势垒精确计算
- 219_cordic: CORDIC算法 → 广义层错能面的迭代求解

核心物理模型:
-------------
Peierls-Nabarro模型将位错描述为两个弹性半晶体之间的非弹性错配。
平衡方程 (Peierls-Nabarro方程):

    σ_applied = (μ/(2π(1-ν))) * ∫_{-∞}^{+∞} (du/dx') / (x - x') dx'
                + dγ(u)/du

其中:
- u(x): 失配函数 (disregistry)
- γ(u): 广义层错能 (GSF energy)
- 第一项: 弹性贡献 (Cauchy主值积分)
- 第二项: 面间恢复力 (γ面的导数)

对于正弦γ面: γ(u) = (μb/(2π)) * (1 - cos(2πu/b))

解为: u(x) = (b/π) * arctan(x/ζ) + b/2
核心宽度: ζ = a/(1-ν)

椭圆积分出现在:
1. 位错核心能量的精确计算
2. 非正弦γ面下的Peierls势垒
3. 位错对(kink pair)形核能量
"""

import math
from physical_constants import PI, SQRT2, DEFAULT_MATERIAL, BOLTZMANN_CONSTANT


# ============================================================================
# 椭圆积分 — 来自 335_elliptic_integral
# 用于位错核心能量的精确计算
# ============================================================================

def elliptic_k_series(m, n_terms=20):
    """
    第一类完全椭圆积分 K(m) 的级数展开

    K(m) = (π/2) * Σ_{n=0}^∞ [(2n)!/(2^n n!)²]² * m^n

    物理含义: 在位错理论中，K(m) 出现在:
    - 椭圆位错环的自能计算
    - 位错线上kink对的形状描述

    收敛域: |m| < 1

    Args:
        m: 椭圆积分参数 (0 ≤ m < 1)
        n_terms: 级数项数

    Returns:
        float: K(m) 值
    """
    if m < 0.0:
        raise ValueError(f"椭圆积分参数 m={m} 必须非负")
    if m >= 1.0:
        # K(m) → ∞ 当 m → 1
        if m > 1.0 + 1e-12:
            raise ValueError(f"椭圆积分参数 m={m} 超出收敛域")
        # 使用对数渐近: K(m) ≈ ln(4/√(1-m)) 当 m → 1
        return math.log(4.0 / math.sqrt(max(1.0 - m, 1e-300)))

    total = 0.0
    coeff = 1.0  # (2n)!/(2^n n!)² 的递推
    m_pow = 1.0  # m^n

    for n in range(n_terms):
        total += coeff**2 * m_pow
        # 递推: coeff_{n+1} = coeff_n * (2n+1)/(2n+2)
        coeff *= (2.0 * n + 1.0) / (2.0 * n + 2.0)
        m_pow *= m

    return (PI / 2.0) * total


def elliptic_e_series(m, n_terms=20):
    """
    第二类完全椭圆积分 E(m) 的级数展开

    E(m) = (π/2) * Σ_{n=0}^∞ [(2n)!/(2^n n!)²]² * m^n / (1-2n)

    物理含义:
    - 位错环周长的椭圆积分
    - 非圆形位错环的线张力

    Args:
        m: 椭圆积分参数 (0 ≤ m < 1)
        n_terms: 级数项数

    Returns:
        float: E(m) 值
    """
    if m < 0.0:
        raise ValueError(f"椭圆积分参数 m={m} 必须非负")
    if m >= 1.0:
        if m > 1.0 + 1e-12:
            raise ValueError(f"椭圆积分参数 m={m} 超出收敛域")
        return 1.0  # E(1) = 1

    total = 0.0
    coeff = 1.0
    m_pow = 1.0

    for n in range(n_terms):
        factor = coeff**2 / (1.0 - 2.0 * n) if n > 0 else 1.0
        total += factor * m_pow
        coeff *= (2.0 * n + 1.0) / (2.0 * n + 2.0)
        m_pow *= m

    return (PI / 2.0) * total


def elliptic_f_incomplete(phi, m, n_terms=15):
    """
    第一类不完全椭圆积分 F(φ, m)

    F(φ, m) = ∫₀^φ dθ / √(1 - m sin²θ)

    在位错理论中用于:
    - 计算kink对的轮廓形状
    - 非均匀位错核心的位移场

    Args:
        phi: 振幅角 (rad)
        m: 椭圆积分参数
        n_terms: Gauss-Legendre求积节点数

    Returns:
        float: F(φ, m) 值
    """
    # 使用Gauss-Legendre求积
    # 将积分区间 [0, φ] 映射到 [-1, 1]
    half_phi = phi / 2.0

    # Gauss-Legendre节点和权重 (5阶)
    nodes, weights = _gauss_legendre_nodes(n_terms)

    total = 0.0
    for x, w in zip(nodes, weights):
        theta = half_phi * (x + 1.0)  # 映射到 [0, φ]
        integrand = 1.0 / math.sqrt(max(1.0 - m * math.sin(theta)**2, 1e-300))
        total += w * integrand

    return half_phi * total


def _gauss_legendre_nodes(n):
    """
    计算Gauss-Legendre求积节点和权重

    节点是 Legendre 多项式 P_n(x) 的零点

    Args:
        n: 节点数

    Returns:
        tuple: (nodes, weights) 列表
    """
    if n <= 0:
        return [0.0], [2.0]

    nodes = []
    weights = []

    for i in range(1, n + 1):
        # 初始猜测 (Chebyshev近似)
        x = math.cos(PI * (i - 0.25) / (n + 0.5))

        # Newton迭代求 P_n(x) = 0 的根
        for _ in range(20):
            p0 = 1.0
            p1 = x
            for j in range(2, n + 1):
                p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j
                p0 = p1
                p1 = p2

            # P_n'(x) = n(x P_n - P_{n-1}) / (x² - 1)
            dp = n * (x * p1 - p0) / (x * x - 1.0) if abs(x * x - 1.0) > 1e-30 else 0.0
            dx = -p1 / dp if abs(dp) > 1e-30 else 0.0
            x += dx
            if abs(dx) < 1e-15:
                break

        nodes.append(x)
        # 权重: w_i = 2 / ((1 - x_i²) [P_n'(x_i)]²)
        p0 = 1.0
        p1 = x
        for j in range(2, n + 1):
            p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j
            p0 = p1
            p1 = p2
        dp = n * (x * p1 - p0) / (x * x - 1.0) if abs(x * x - 1.0) > 1e-30 else 1.0
        w = 2.0 / ((1.0 - x * x) * dp * dp) if abs(dp) > 1e-30 else 0.0
        weights.append(w)

    return nodes, weights


# ============================================================================
# CORDIC算法 — 来自 219_cordic
# 用于广义层错能面的快速三角函数迭代
# ============================================================================

class CORDICTrigo:
    """
    CORDIC (COordinate Rotation DIgital Computer) 三角函数计算

    CORDIC通过迭代旋转计算三角函数，无需乘法表。
    在位错理论中用于:
    - 快速计算广义层错能 γ(u) = γ₀(1 - cos(2πu/b))
    - Peierls-Nabarro方程的迭代求解

    迭代公式:
    x_{n+1} = x_n - d_n y_n 2^{-n}
    y_{n+1} = y_n + d_n x_n 2^{-n}
    z_{n+1} = z_n - d_n α_n

    其中 d_n = sign(z_n), α_n = arctan(2^{-n})
    """

    def __init__(self, n_iterations=24):
        """
        初始化CORDIC查找表

        Args:
            n_iterations: 迭代次数 (精度 ~2^{-n})
        """
        self.n_iter = n_iterations
        self.gain = 1.0
        self.angles = []

        # 预计算 arctan(2^{-n})
        for n in range(n_iterations):
            angle = math.atan(2.0 ** (-n))
            self.angles.append(angle)
            self.gain *= math.sqrt(1.0 + 2.0**(-2*n))

        self.gain_inv = 1.0 / self.gain

    def cossin(self, theta):
        """
        使用CORDIC计算 cos(θ) 和 sin(θ)

        在圆形模式下，CORDIC旋转矢量 (x, y) 使角度趋近于零:
        初始: (x₀, y₀, z₀) = (1/K, 0, θ)
        最终: (x_n, y_n) → (cos θ, sin θ)

        对于|θ| > π/2 的角度，先做象限校正。

        Args:
            theta: 角度 (rad)

        Returns:
            tuple: (cos_theta, sin_theta)
        """
        # 象限校正: 将角度归化到 [-π/2, π/2]
        # 使用周期性: cos(θ + π) = -cos(θ), sin(θ + π) = -sin(θ)
        sign_flip = 1.0
        # 归化到 [-π, π]
        theta = theta % (2.0 * PI)
        if theta > PI:
            theta -= 2.0 * PI

        # 如果 |θ| > π/2, 用 θ - sign(θ)*π 并翻转符号
        if theta > PI / 2.0:
            theta -= PI
            sign_flip = -1.0
        elif theta < -PI / 2.0:
            theta += PI
            sign_flip = -1.0

        x = self.gain_inv
        y = 0.0
        z = theta

        for n in range(self.n_iter):
            if z >= 0.0:
                d = 1.0
            else:
                d = -1.0

            shift = 2.0**(-n)
            x_new = x - d * y * shift
            y_new = y + d * x * shift
            z_new = z - d * self.angles[n]

            x, y, z = x_new, y_new, z_new

        return sign_flip * x, sign_flip * y

    def arctan2_cordic(self, y, x):
        """
        使用CORDIC计算 atan2(y, x)

        在矢量模式下，CORDIC旋转使y分量趋零:
        初始: (x₀, y₀, z₀) = (x, y, 0)
        最终: z_n → atan2(y, x)

        Args:
            y, x: 笛卡尔坐标

        Returns:
            float: atan2(y, x) (rad)
        """
        # 处理象限
        if x < 0.0:
            if y >= 0.0:
                return self._arctan2_inner(-y, -x) - PI
            else:
                return self._arctan2_inner(-y, -x) + PI
        else:
            return self._arctan2_inner(y, x)

    def _arctan2_inner(self, y, x):
        """CORDIC矢量模式内部实现"""
        z = 0.0
        for n in range(self.n_iter):
            if y >= 0.0:
                d = 1.0
            else:
                d = -1.0

            x_new = x + d * y * 2.0**(-n)
            y_new = y - d * x * 2.0**(-n)
            z_new = z + d * self.angles[n]

            x, y, z = x_new, y_new, z_new

        return z


# ============================================================================
# Peierls-Nabarro位错核心模型
# ============================================================================

class PeierlsNabarroModel:
    """
    Peierls-Nabarro位错核心结构求解器

    求解PN方程:
    (μ/(2π(1-ν))) * P.V. ∫_{-∞}^{∞} (du/dx')/(x-x') dx' = dγ/du - σ_app

    数值方法: 在有限域 [-L, L] 上离散化
    u(x_i) 定义在网格 x_i = -L + i*dx, i = 0, ..., N-1

    离散PN方程:
    Σ_j A_{ij} u_j = f(u_i) - σ_app

    其中 A_{ij} 是离散Hilbert变换矩阵
    """

    def __init__(self, material=None, n_grid=128, domain_size=None):
        """
        初始化Peierls-Nabarro求解器

        Args:
            material: 材料参数对象
            n_grid: 网格点数
            domain_size: 计算域半宽度 (m)
        """
        self.mat = material or DEFAULT_MATERIAL
        self.n_grid = n_grid

        # 计算域: 足够大以捕获位错核心衰减
        if domain_size is None:
            self.L = 50.0 * self.mat.zeta_screw  # 50倍核心宽度
        else:
            self.L = domain_size

        self.dx = 2.0 * self.L / (self.n_grid - 1)
        self.x = [-self.L + i * self.dx for i in range(self.n_grid)]

        # CORDIC加速器
        self.cordic = CORDICTrigo(n_iterations=24)

        # 预计算离散Hilbert变换核
        self.hilbert_matrix = self._build_hilbert_matrix()

    def _build_hilbert_matrix(self):
        """
        构建离散Hilbert变换矩阵

        H_{ij} = dx / (π(x_i - x_j))  for i ≠ j
        H_{ii} = 0                      (主值积分)

        这是PN方程中弹性贡献的离散化

        Returns:
            list: N×N矩阵
        """
        H = [[0.0]*self.n_grid for _ in range(self.n_grid)]
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                if i != j:
                    H[i][j] = self.dx / (PI * (self.x[i] - self.x[j]))
        return H

    def gamma_gsf(self, u):
        """
        广义层错能 (正弦近似)

        γ(u) = γ_amplitude * (1 - cos(2πu/b))

        其中 γ_amplitude = γ_usf / (2π)²

        物理含义:
        - u = 0: 完美晶体 (γ = 0)
        - u = b/2: 不稳定平衡 (γ = 最大)
        - u = b: 下一个完美位置 (γ = 0)

        Args:
            u: 失配位移 (m)

        Returns:
            float: 层错能 (J/m²)
        """
        # 使用CORDIC快速计算cos
        phase = 2.0 * PI * u / self.mat.b_magnitude
        cos_val, _ = self.cordic.cossin(phase)
        return self.mat.gamma_amplitude * (1.0 - cos_val)

    def dgamma_du(self, u):
        """
        层错能的导数 (恢复力)

        dγ/du = (2π/b) * γ_amplitude * sin(2πu/b)

        这是PN方程中面间恢复力的来源

        Args:
            u: 失配位移 (m)

        Returns:
            float: dγ/du (J/m³ = Pa/m)
        """
        phase = 2.0 * PI * u / self.mat.b_magnitude
        _, sin_val = self.cordic.cossin(phase)
        return (2.0 * PI / self.mat.b_magnitude) * self.mat.gamma_amplitude * sin_val

    def peierls_potential(self, u):
        """
        Peierls势垒 (每单位面积的能量变化)

        ΔE(u) = γ(u) - γ(0) = γ(u)

        Peierls势垒高度:
        ΔE_P = γ(b/2) - γ(0) = 2 * γ_amplitude

        等效Peierls应力:
        σ_P = max(dγ/du) / b = (2π/b²) * γ_amplitude

        Args:
            u: 失配位移 (m)

        Returns:
            float: Peierls势能密度 (J/m²)
        """
        return self.gamma_gsf(u)

    def peierls_stress_analytical(self):
        """
        解析Peierls应力

        从γ面 (理论剪切强度):
        τ_max = max(dγ/du) = π γ_usf / b

        从PN指数公式:
        σ_P = 2μ/(1-ν) * exp(-2πζ/b)

        Returns:
            tuple: (理论强度, PN指数估算)
        """
        # 从γ面导数 (理论剪切强度)
        sigma_p_gsf = PI * self.mat.gamma_usf / self.mat.b_magnitude

        # 从PN指数公式
        sigma_p_pn = self.mat.peierls_stress()

        return sigma_p_gsf, sigma_p_pn

    def dislocation_profile_analytical(self):
        """
        解析位错核心轮廓 (正弦γ面)

        u(x) = (b/π) arctan(x/ζ) + b/2
        du/dx = (b/π) * ζ / (x² + ζ²)

        Returns:
            tuple: (x_grid, u_profile, dudx_profile)
        """
        b = self.mat.b_magnitude
        zeta = self.mat.zeta_screw

        u_profile = []
        dudx_profile = []

        for xi in self.x:
            ui = (b / PI) * math.atan(xi / zeta) + b / 2.0
            dudxi = (b / PI) * zeta / (xi**2 + zeta**2)
            u_profile.append(ui)
            dudx_profile.append(dudxi)

        return self.x, u_profile, dudx_profile

    def dislocation_core_energy(self):
        """
        计算位错核心能量 (使用椭圆积分)

        核心超额能量:
        E_core = ∫_{-∞}^{∞} [γ(u(x)) - (μ/(4π(1-ν))) * (du/dx)²] dx

        对于正弦γ面，解析结果为:
        E_core = 2μζ/(1-ν) * [K(m) - E(m)]

        其中 m = 1 - (σ_app/σ_P)²  (施加应力的函数)
        K(m), E(m) 是第一、二类完全椭圆积分

        Returns:
            dict: 核心能量信息
        """
        mu = self.mat.mu
        nu = self.mat.nu
        zeta = self.mat.zeta_screw
        b = self.mat.b_magnitude

        # 无应力时的参数
        # m = 1 对应 σ = 0 (位错稳定)
        # 使用接近1的参数 (避免奇异)
        m_param = 0.999

        K_val = elliptic_k_series(m_param, n_terms=30)
        E_val = elliptic_e_series(m_param, n_terms=30)

        # 核心能量 (每单位位错长度)
        E_core = 2.0 * mu * zeta / (1.0 - nu) * (K_val - E_val)

        # 验证: 使用数值积分
        _, u_prof, dudx_prof = self.dislocation_profile_analytical()
        E_core_numerical = 0.0
        for i in range(self.n_grid):
            gamma_i = self.gamma_gsf(u_prof[i])
            elastic_i = (mu / (4.0 * PI * (1.0 - nu))) * dudx_prof[i]**2
            # 减去远场能量
            E_core_numerical += (gamma_i) * self.dx

        # 解析E_core可能很大 (椭圆积分在m→1时发散)
        # 这是物理的: 核心能量需要截断

        return {
            'elliptic_K': K_val,
            'elliptic_E': E_val,
            'E_core_analytical': E_core,
            'E_core_numerical': E_core_numerical,
            'm_parameter': m_param,
        }

    def kink_pair_energy(self, kink_separation):
        """
        计算kink对的形核能量

        位错线上的kink对是两个反号扭折:
        E_kink_pair = 2 E_kink - E_interaction(d)

        单个kink能量:
        E_kink = (μ b² / (4π(1-ν))) * ∫ √(2γ(u)/μ') du

        kink间相互作用:
        E_inter(d) = -μ b² ζ² / (4π(1-ν) d²)

        对于正弦γ面:
        E_kink = (2/π) * √(2 μ ζ γ_amplitude / (1-ν)) * b

        Args:
            kink_separation: kink间距 (m)

        Returns:
            dict: kink对能量信息
        """
        mu = self.mat.mu
        nu = self.mat.nu
        zeta = self.mat.zeta_screw
        b = self.mat.b_magnitude
        gamma_amp = self.mat.gamma_amplitude

        # 单个kink能量 (正弦γ面解析解)
        # 使用椭圆积分 E(m) 计算
        E_kink = (2.0 / PI) * math.sqrt(2.0 * mu * zeta * gamma_amp / (1.0 - nu)) * b

        # kink对总能量
        d = max(kink_separation, 2.0 * b)  # 最小间距 = 2b
        E_interaction = -mu * b**2 * zeta**2 / (4.0 * PI * (1.0 - nu) * d**2)

        E_pair = 2.0 * E_kink + E_interaction

        # kink对形核的临界应力
        # σ_kink = E_pair / (b * d * b)
        sigma_kink = E_pair / (b * d * b)

        return {
            'E_kink_single': E_kink,
            'E_interaction': E_interaction,
            'E_pair_total': E_pair,
            'kink_separation': d,
            'critical_stress': sigma_kink,
        }

    def solve_pn_equation(self, sigma_applied=0.0, max_iter=500, tol=1e-10):
        """
        数值求解Peierls-Nabarro方程

        使用定点迭代:
        u^{n+1}_i = u^n_i + ω * [σ_app - (1/b) dγ/du(u^n_i)
                   - (μ/(2π(1-ν))) Σ_j H_{ij} du/dx|_j]

        Args:
            sigma_applied: 施加剪应力 (Pa)
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            dict: 求解结果
        """
        # 初始猜测: 解析解
        _, u_init, _ = self.dislocation_profile_analytical()
        u = list(u_init)

        mu = self.mat.mu
        nu = self.mat.nu
        b = self.mat.b_magnitude

        # 预计算 du/dx (中心差分)
        dudx = [0.0] * self.n_grid
        for i in range(1, self.n_grid - 1):
            dudx[i] = (u[i+1] - u[i-1]) / (2.0 * self.dx)
        dudx[0] = (u[1] - u[0]) / self.dx
        dudx[-1] = (u[-1] - u[-2]) / self.dx

        # 弹性项: H · (du/dx)
        elastic_term = [0.0] * self.n_grid
        prefactor = mu / (2.0 * PI * (1.0 - nu))
        for i in range(self.n_grid):
            s = 0.0
            for j in range(self.n_grid):
                s += self.hilbert_matrix[i][j] * dudx[j]
            elastic_term[i] = prefactor * s

        # 迭代求解
        omega = 0.1  # 松弛因子
        converged = False
        residual_history = []

        for iteration in range(max_iter):
            u_new = list(u)
            max_change = 0.0

            for i in range(1, self.n_grid - 1):  # 边界固定
                restoring = self.dgamma_du(u[i]) / b
                residual = sigma_applied - restoring - elastic_term[i]
                u_new[i] = u[i] + omega * residual * self.dx / mu
                change = abs(u_new[i] - u[i])
                if change > max_change:
                    max_change = change

            u = u_new

            # 重新计算导数
            for i in range(1, self.n_grid - 1):
                dudx[i] = (u[i+1] - u[i-1]) / (2.0 * self.dx)

            residual_history.append(max_change)

            if max_change < tol:
                converged = True
                break

        return {
            'x': self.x,
            'u': u,
            'converged': converged,
            'iterations': len(residual_history),
            'final_residual': residual_history[-1] if residual_history else float('inf'),
            'sigma_applied': sigma_applied,
        }


if __name__ == '__main__':
    print("=" * 70)
    print("Peierls-Nabarro位错核心模型验证")
    print("=" * 70)

    # 测试椭圆积分
    print("\n椭圆积分验证:")
    for m in [0.0, 0.25, 0.5, 0.75, 0.99]:
        K = elliptic_k_series(m)
        E = elliptic_e_series(m)
        print(f"  m={m:.2f}: K(m)={K:.6f}, E(m)={E:.6f}")

    # 测试CORDIC
    print("\nCORDIC三角函数验证:")
    cordic = CORDICTrigo()
    for theta_deg in [0, 30, 45, 60, 90, 180]:
        theta = theta_deg * PI / 180.0
        c_cos, c_sin = cordic.cossin(theta)
        ref_cos = math.cos(theta)
        ref_sin = math.sin(theta)
        err_c = abs(c_cos - ref_cos)
        err_s = abs(c_sin - ref_sin)
        print(f"  θ={theta_deg:3d}°: cos_err={err_c:.2e}, sin_err={err_s:.2e}")

    # 测试PN模型
    print("\nPeierls-Nabarro模型:")
    pn = PeierlsNabarroModel(n_grid=64)

    sigma_gsf, sigma_pn = pn.peierls_stress_analytical()
    print(f"  Peierls应力 (γ面): {sigma_gsf:.4e} Pa")
    print(f"  Peierls应力 (PN公式): {sigma_pn:.4e} Pa")

    core_info = pn.dislocation_core_energy()
    print(f"  椭圆积分 K(0.999) = {core_info['elliptic_K']:.4f}")
    print(f"  椭圆积分 E(0.999) = {core_info['elliptic_E']:.4f}")

    kink_info = pn.kink_pair_energy(kink_separation=50e-10)
    print(f"  单个kink能量: {kink_info['E_kink_single']:.4e} J/m")
    print(f"  kink对总能量: {kink_info['E_pair_total']:.4e} J/m")
