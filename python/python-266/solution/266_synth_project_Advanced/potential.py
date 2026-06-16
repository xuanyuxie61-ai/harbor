"""
potential.py — 周期势场构造与 Fourier 分析
=============================================

本模块实现 1D 晶体中各类周期势场的构造, 包括:

  - 外部周期势 V_ext(x) (Mathieu 型, Kronig-Penney 型, 赝势型)
  - Hartree 势 V_H(x) (源自 电子-电子库仑相互作用)
  - 交换关联势 V_xc(x) (LDA/PBE 近似)

融合种子项目:
  - 549_humps: Lorentzian 型赝势峰, 带有解析导数
  - 596_interp_trig: 三角插值, 用于周期函数的光滑表示
  - 302_disk01_rule: Gauss-Legendre 求积节点, 用于势能积分
  - 271_dg1d_advection: Vandermonde 矩阵与 Jacobi 多项式,
    用于势能的谱展开

核心物理:
  有效势: V_eff(x) = V_ext(x) + V_H(x) + V_xc(x)
  其中:
    V_ext(x) = Σ_n V_n cos(n·G·x), G = 2π/a  (Fourier 展开)
    V_H(x) = ∫ n(x')/|x-x'| dx'              (Hartree, Poisson 方程)
    V_xc(x) = δE_xc/δn(x)                    (交换关联势)

  Mathieu 势模型 (本项目的标准测试):
    V(x) = V₁ cos(Gx) + V₂ cos(2Gx) + V₃ cos(3Gx)
  该势场在 k = G/2 处打开能隙, 能隙宽度 ≈ |V_n|。
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict
from physical_constants import PI, TWO_PI, FOUR_PI


# ============================================================
# 势场基类
# ============================================================

class PeriodicPotential:
    """
    1D 周期势场的抽象基类。

    所有势场必须实现:
      V(x): 势能值
      dV(x): 一阶导数 (用于 Ehrenfest 力和 Hellmann-Feynman 定理)
      d2V(x): 二阶导数 (用于稳定性分析)
      fourier_coefficients(n_max): Fourier 展开系数 V_n
    """

    def __init__(self, a: float):
        """
        Parameters
        ----------
        a : float
            晶格常数
        """
        self.a = a
        self.G = TWO_PI / a  # 第一倒格矢

    def V(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dV(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def d2V(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fourier_coefficients(self, n_max: int) -> np.ndarray:
        """
        计算 Fourier 系数:
          V_n = (1/a) ∫₀ᵃ V(x) exp(-i·n·G·x) dx

        对于实势场, V_{-n} = V_n* = V_n (实数)。

        Parameters
        ----------
        n_max : int
            最大 Fourier 指标

        Returns
        -------
        coeffs : np.ndarray, shape (2*n_max+1,)
            V_n for n = -n_max, ..., 0, ..., n_max
        """
        n_grid = 1024
        x, dx = np.linspace(0, self.a, n_grid, endpoint=False), self.a / n_grid
        Vx = self.V(x)

        coeffs = np.zeros(2 * n_max + 1, dtype=complex)
        for i, n in enumerate(range(-n_max, n_max + 1)):
            integrand = Vx * np.exp(-1j * n * self.G * x)
            coeffs[i] = np.sum(integrand) * dx / self.a

        return coeffs.real  # 实势场


# ============================================================
# Mathieu 型势场 (标准测试模型)
# ============================================================

class MathieuPotential(PeriodicPotential):
    """
    Mathieu 型周期势:
      V(x) = V₁ cos(Gx) + V₂ cos(2Gx) + V₃ cos(3Gx)

    其中 G = 2π/a 为第一倒格矢。

    物理意义:
    - V₁ 控制第一 Brillouin 区边界 (k=±π/a) 的主能隙
    - V₂ 控制 Γ-X 之间的能带曲率修正
    - V₃ 控制更高阶的能带折叠效应

    Mathieu 方程标准形式:
      -d²ψ/dx² + [2q₁ cos(2x) + 2q₂ cos(4x)] ψ = λψ
    对应参数: q₁ = V₁·a²/(4π²), q₂ = V₂·a²/(16π²)

    导数:
      V'(x) = -G[V₁ sin(Gx) + 2V₂ sin(2Gx) + 3V₃ sin(3Gx)]
      V''(x) = -G²[V₁ cos(Gx) + 4V₂ cos(2Gx) + 9V₃ cos(3Gx)]
    """

    def __init__(self, a: float, V1: float = 0.5,
                 V2: float = 0.15, V3: float = 0.03):
        super().__init__(a)
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3

    def V(self, x: np.ndarray) -> np.ndarray:
        Gx = self.G * x
        return (self.V1 * np.cos(Gx) +
                self.V2 * np.cos(2.0 * Gx) +
                self.V3 * np.cos(3.0 * Gx))

    def dV(self, x: np.ndarray) -> np.ndarray:
        Gx = self.G * x
        return -self.G * (self.V1 * np.sin(Gx) +
                          2.0 * self.V2 * np.sin(2.0 * Gx) +
                          3.0 * self.V3 * np.sin(3.0 * Gx))

    def d2V(self, x: np.ndarray) -> np.ndarray:
        Gx = self.G * x
        return -(self.G ** 2) * (self.V1 * np.cos(Gx) +
                                  4.0 * self.V2 * np.cos(2.0 * Gx) +
                                  9.0 * self.V3 * np.cos(3.0 * Gx))

    def mathieu_parameters(self) -> Dict[str, float]:
        """返回标准 Mathieu 方程参数 q₁, q₂"""
        return {
            'q1': self.V1 * self.a ** 2 / (4.0 * PI ** 2),
            'q2': self.V2 * self.a ** 2 / (16.0 * PI ** 2),
        }


# ============================================================
# 赝势 (源自 549_humps: Lorentzian 型峰)
# ============================================================

class PseudoPotential(PeriodicPotential):
    """
    基于 Lorentzian 峰的光滑赝势 (融合 549_humps):

      V(x) = Σ_j A_j / [(x - x_j)² + σ²] - V_offset

    周期化通过镜像求和实现:
      V_periodic(x) = Σ_{n=-N_image}^{N_image} V(x + n·a)

    这种赝势在固态 DFT 中用于替代奇异库仑势,
    使得实空间离散化更加稳定。

    解析导数 (与 humps 函数一致):
      V'(x) = -2x · A / [(x² + σ²)²]
      V''(x) = A · [6x² - 2σ²] / [(x² + σ²)³]

    参数:
      A_j: 峰强度 (对应离子赝势的价电子耦合)
      x_j: 原子位置
      σ: 赝势半径 (越小越接近真实库仑势)
    """

    def __init__(self, a: float, positions: np.ndarray,
                 strengths: np.ndarray, sigma: float = 0.3,
                 n_images: int = 5):
        """
        Parameters
        ----------
        a : float
            晶格常数
        positions : np.ndarray
            原子位置 (在 [0, a) 内)
        strengths : np.ndarray
            各原子赝势强度
        sigma : float
            赝势半径
        n_images : int
            镜像求和范围
        """
        super().__init__(a)
        self.positions = np.asarray(positions)
        self.strengths = np.asarray(strengths)
        self.sigma = sigma
        self.n_images = n_images

        if len(self.positions) != len(self.strengths):
            raise ValueError("positions 和 strengths 长度必须相同")
        if sigma <= 0:
            raise ValueError(f"赝势半径必须 > 0, 得到 σ={sigma}")

    def _single_lorentzian(self, x: np.ndarray, center: float,
                           strength: float) -> np.ndarray:
        """单个 Lorentzian 峰: A / [(x-x₀)² + σ²]"""
        dx = x - center
        return strength / (dx ** 2 + self.sigma ** 2)

    def _single_d1(self, x: np.ndarray, center: float,
                   strength: float) -> np.ndarray:
        """Lorentzian 一阶导数"""
        dx = x - center
        s2 = self.sigma ** 2
        return -2.0 * strength * dx / (dx ** 2 + s2) ** 2

    def _single_d2(self, x: np.ndarray, center: float,
                   strength: float) -> np.ndarray:
        """Lorentzian 二阶导数"""
        dx = x - center
        s2 = self.sigma ** 2
        dx2 = dx ** 2
        return strength * (6.0 * dx2 - 2.0 * s2) / (dx2 + s2) ** 3

    def V(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for n_img in range(-self.n_images, self.n_images + 1):
            shift = n_img * self.a
            for pos, amp in zip(self.positions, self.strengths):
                result += self._single_lorentzian(x, pos + shift, amp)
        return result

    def dV(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for n_img in range(-self.n_images, self.n_images + 1):
            shift = n_img * self.a
            for pos, amp in zip(self.positions, self.strengths):
                result += self._single_d1(x, pos + shift, amp)
        return result

    def d2V(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for n_img in range(-self.n_images, self.n_images + 1):
            shift = n_img * self.a
            for pos, amp in zip(self.positions, self.strengths):
                result += self._single_d2(x, pos + shift, amp)
        return result


# ============================================================
# 三角插值势场 (源自 596_interp_trig)
# ============================================================

class TrigInterpolatedPotential(PeriodicPotential):
    """
    使用三角插值构造的周期势场 (融合 596_interp_trig)。

    给定 N 个等距节点上的势场值 V(x_j), 构造
    三角插值函数:

    当 N 为奇数:
      V_interp(x) = Σ_j V(x_j) · L_j(x)
    其中 L_j(x) = sin(N·π·(x-x_j)/a) / [N·sin(π·(x-x_j)/a)]

    当 N 为偶数:
      L_j(x) = sin(N·π·(x-x_j)/a) / [N·tan(π·(x-x_j)/a)]

    这保证了:
    1. 严格的周期性 (自动满足 Bloch 定理的周期部分)
    2. 指数收敛 (对于解析势场)
    3. 无 Gibbs 现象

    物理应用: 从粗网格 DFT 计算结果插值到细网格,
    或从离散 Fourier 系数重建连续势场。
    """

    def __init__(self, a: float, x_data: np.ndarray,
                 V_data: np.ndarray):
        """
        Parameters
        ----------
        a : float
            晶格常数 (周期)
        x_data : np.ndarray
            等距节点坐标
        V_data : np.ndarray
            节点上的势场值
        """
        super().__init__(a)
        self.x_data = np.asarray(x_data)
        self.V_data = np.asarray(V_data)
        self.n_data = len(x_data)
        self.h = a / self.n_data  # 节点间距

        if len(self.V_data) != self.n_data:
            raise ValueError("x_data 和 V_data 长度不匹配")

    def _cardinal(self, x: np.ndarray, xj: float) -> np.ndarray:
        """
        三角基数函数 (源自 596_interp_trig)。

        N 奇数:
          L(x, x_j) = sin(π(x-x_j)/h) / [N·sin(π(x-x_j)/(Nh))]

        N 偶数:
          L(x, x_j) = sin(π(x-x_j)/h) / [N·tan(π(x-x_j)/(Nh))]
        """
        dx = x - xj
        arg = PI * dx / self.h

        # 避免除零
        small = np.abs(dx) < 1e-14
        result = np.zeros_like(x, dtype=float)

        if self.n_data % 2 == 1:
            # 奇数: sin(N·arg/N) / (N·sin(arg/N))
            denom = self.n_data * np.sin(arg / self.n_data)
            safe = np.abs(denom) > 1e-14
            result[safe] = np.sin(arg[safe]) / denom[safe]
            result[~safe] = 1.0  # L(x_j, x_j) = 1
        else:
            # 偶数: sin(arg) / (N·tan(arg/N))
            denom = self.n_data * np.tan(arg / self.n_data)
            safe = np.abs(denom) > 1e-14
            result[safe] = np.sin(arg[safe]) / denom[safe]
            result[~safe] = 1.0

        result[small] = 1.0
        return result

    def V(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x, dtype=float)
        for j in range(self.n_data):
            result += self.V_data[j] * self._cardinal(x, self.x_data[j])
        return result

    def dV(self, x: np.ndarray) -> np.ndarray:
        """
        三角插值势的一阶导数 (数值差分)。
        使用中心差分, 步长 h/100。
        """
        eps = self.h / 100.0
        return (self.V(x + eps) - self.V(x - eps)) / (2.0 * eps)

    def d2V(self, x: np.ndarray) -> np.ndarray:
        eps = self.h / 100.0
        return (self.V(x + eps) - 2.0 * self.V(x) + self.V(x - eps)) / (eps ** 2)


# ============================================================
# Hartree 势 (源自 362_fd1d_heat_steady)
# ============================================================

def compute_hartree_potential(density: np.ndarray, dx: float,
                               a: float,
                               strength: float = 1.0) -> np.ndarray:
    """
    计算 1D Hartree 势 (融合 362_fd1d_heat_steady 的 Poisson 求解器)。

    在 1D 中, Hartree 势满足 Poisson 方程:
      -d²V_H/dx² = 4π·n(x)    (Gauss 单位制)

    或使用有效强度参数:
      -d²V_H/dx² = 4π·α·n(x)

    其中 α 为 Hartree 势强度参数 (控制 e-e 相互作用的强度)。

    边界条件: 周期性 (V_H(0) = V_H(a), V_H'(0) = V_H'(a))

    在周期性边界条件下, Poisson 方程要求:
      ∫₀ᵃ n(x) dx = N_e / a (平均密度)

    解法: Fourier 空间求解
      V_H(G_n) = 4π·α·n(G_n) / G_n²   for n ≠ 0
      V_H(0) = 0 (jellium 背景抵消)

    等价于 362_fd1d_heat_steady 中的三对角系统求解:
      -d/dx(k(x) dV/dx) = f(x)
    其中 k(x) = 1 (均匀介质), f(x) = -4π·α·n(x)。

    Parameters
    ----------
    density : np.ndarray, shape (N,)
        电子密度 n(x_i)
    dx : float
        网格间距
    a : float
        原胞长度
    strength : float
        Hartree 耦合常数 α

    Returns
    -------
    V_H : np.ndarray, shape (N,)
        Hartree 势
    """
    N = len(density)

    # Fourier 空间方法
    n_fourier = np.fft.fft(density)
    freqs = np.fft.fftfreq(N, d=dx)

    V_H_fourier = np.zeros_like(n_fourier)
    for i in range(N):
        if i == 0:
            V_H_fourier[i] = 0.0  # 零频分量 (jellium 背景)
        else:
            G = TWO_PI * freqs[i]
            V_H_fourier[i] = FOUR_PI * strength * n_fourier[i] / (G ** 2)

    V_H = np.fft.ifft(V_H_fourier).real
    return V_H


# ============================================================
# 势场 Fourier 分析 (源自 302_disk01_rule / 271_dg1d_advection)
# ============================================================

def fourier_analysis_potential(potential: PeriodicPotential,
                                n_max: int,
                                n_quad: int = 512
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    对周期势场进行 Fourier 分析。

    使用 Gauss-Legendre 求积 (源自 302_disk01_rule 的
    legendre_ek_compute) 计算 Fourier 系数:

      V_n = (1/a) ∫₀ᵃ V(x) exp(-i·n·G·x) dx

    对于解析势场, Fourier 系数的衰减速度反映势场的光滑性:
    - C∞ 势场: V_n ~ O(n^{-k}) 对任意 k
    - 解析势场: V_n ~ O(e^{-cn})
    - C^k 势场: V_n ~ O(n^{-k-1})

    Parameters
    ----------
    potential : PeriodicPotential
        势场对象
    n_max : int
        最大 Fourier 指标
    n_quad : int
        求积节点数

    Returns
    -------
    n_indices : np.ndarray
        Fourier 指标 n
    V_n : np.ndarray
        Fourier 系数 V_n (实部, 对实势场)
    """
    a = potential.a

    # 使用 Gauss-Legendre 求积 (源自 302_disk01_rule)
    # 在 [0, a] 上的 N 点 Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
    # 变换到 [0, a]
    x_gl = 0.5 * a * (x_gl + 1.0)
    w_gl = 0.5 * a * w_gl

    Vx = potential.V(x_gl)

    n_indices = np.arange(-n_max, n_max + 1)
    G = TWO_PI / a
    V_n = np.zeros(2 * n_max + 1)

    for i, n in enumerate(n_indices):
        integrand = Vx * np.cos(n * G * x_gl)
        V_n[i] = np.sum(w_gl * integrand) / a

    return n_indices, V_n


# ============================================================
# 势能积分 (源自 681_line_integrals 和 683_line_monte_carlo)
# ============================================================

def potential_energy_integral(potential: PeriodicPotential,
                               density: np.ndarray,
                               x_grid: np.ndarray,
                               dx: float) -> float:
    """
    计算势能-密度积分 (源自 681_line_integrals 的积分方法):

      E_pot = ∫₀ᵃ V(x) · n(x) dx

    使用梯形法则 (周期性函数的梯形法则 = Gauss 求积)。

    对于周期函数 f(x), 梯形法则的误差为:
      E_trap = -a/(12N²) f''(ξ) + O(N^{-4})

    当 f ∈ C∞ 时, 收敛速度为指数级。

    Parameters
    ----------
    potential : PeriodicPotential
        势场
    density : np.ndarray
        电子密度
    x_grid : np.ndarray
        网格坐标
    dx : float
        网格间距

    Returns
    -------
    energy : float
        势能积分值
    """
    Vx = potential.V(x_grid)
    integrand = Vx * density
    # 周期性梯形法则
    energy = np.sum(integrand) * dx
    return energy
