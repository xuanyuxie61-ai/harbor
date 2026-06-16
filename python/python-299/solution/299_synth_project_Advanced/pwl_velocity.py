"""
pwl_velocity.py — 速度空间分段线性插值与乘积求积
=================================================

种子项目映射:
  - 928_pwl_interp_2d_scattered: 2D 散乱数据分段线性插值
  - 919_product_rule: 多维乘积求积规则
  - 937_pyramid_witherden_rule: 金字塔域高阶求积

物理应用:
  1. 在速度网格上重建分布函数 (用于通量计算)
  2. 计算速度空间矩 (密度, 温度, 热流)
  3. Rosenbluth 势的数值积分

PWL 插值原理 (映射自 pwl_interp_2d_scattered):
  原项目: 在 Delaunay 三角网格上做分段线性插值
          zi = α z₁ + β z₂ + γ z₃ (重心坐标)
  本项目: 1D 速度网格上的分段线性插值
          f(v) = f_j (v_{j+1} - v)/(v_{j+1} - v_j) + f_{j+1} (v - v_j)/(...)
          (是 2D 三角插值的 1D 退化形式)

乘积求积原理 (映射自 product_rule):
  原项目: 多维求积规则 = 一维规则的张量积
          w_{i,j,k} = w_i^x · w_j^y · w_k^z
  本项目: 3D 速度空间中的球面积分 = 径向积分 × 角度积分
          ∫ f(v) d³v = 4π ∫₀^∞ f(v) v² dv
          其中 4π 来自角度部分的精确积分 (球谐 Y₀₀)

金字塔求积 (映射自 pyramid_witherden_rule):
  原项目: 方底金字塔上的高精度求积规则 (Witherden & Vincent 2015)
  本项目: 速度-时间锥域上的积分 (用于时空有限元)
          {(v,t) : 0 ≤ t ≤ T, 0 ≤ v ≤ v_max(t)}
          在 (v,t) 空间中是三角形/金字塔形区域
"""

import math
import numpy as np
from scipy import special as sp

from physical_constants import PI, SQRT_PI, FOUR_PI


# ===========================================================================
#  §1  分段线性插值  (源自 pwl_interp_2d_scattered)
# ===========================================================================
class PWLVelocityInterpolator:
    """速度空间分段线性插值器.

    映射自 pwl_interp_2d_scattered_value:
      原项目: 在 2D Delaunay 三角网格上, 对查询点找所在三角形,
              然后用重心坐标 (α, β, γ) 线性插值:
              zi = α z₁ + β z₂ + γ z₃
      本项目: 1D 速度网格上的分段线性插值.
              对查询速度 v, 找到所在区间 [v_j, v_{j+1}],
              然后: f(v) = f_j · (1-t) + f_{j+1} · t
              其中 t = (v - v_j) / (v_{j+1} - v_j)

    物理意义: 保证插值函数 C⁰ 连续, 不会产生非物理振荡.
    在 Fokker-Planck 求解中, 用于在粗细网格之间传递分布函数.
    """

    def __init__(self, v_grid, f_values):
        """
        Parameters
        ----------
        v_grid : ndarray(N)  速度网格 (必须单调递增)
        f_values : ndarray(N)  网格点上的函数值
        """
        self.v_grid = np.asarray(v_grid, dtype=np.float64)
        self.f_values = np.asarray(f_values, dtype=np.float64)
        self.N = len(v_grid)

        # 验证单调性
        if self.N > 1 and np.any(np.diff(self.v_grid) <= 0):
            raise ValueError("速度网格必须严格单调递增")

    def __call__(self, v_query):
        """在查询点 v_query 处计算分段线性插值.

        等价于三角搜索 + 重心坐标的 1D 简化版.
        """
        v_query = np.asarray(v_query, dtype=np.float64)
        scalar_input = v_query.ndim == 0
        v_query = np.atleast_1d(v_query)

        # 找到每个查询点所在的区间
        indices = np.searchsorted(self.v_grid, v_query, side='right') - 1
        indices = np.clip(indices, 0, self.N - 2)

        # 计算局部参数 t
        v_left = self.v_grid[indices]
        v_right = self.v_grid[indices + 1]
        dv = v_right - v_left
        dv = np.where(dv > 1e-30, dv, 1e-30)  # 避免除零
        t = (v_query - v_left) / dv

        # 分段线性插值 (1D 重心坐标)
        f_left = self.f_values[indices]
        f_right = self.f_values[indices + 1]
        result = f_left * (1.0 - t) + f_right * t

        # 边界外推: 对于超出范围的查询, 使用最近的边界值
        result = np.where(v_query <= self.v_grid[0], self.f_values[0], result)
        result = np.where(v_query >= self.v_grid[-1], self.f_values[-1], result)

        return float(result[0]) if scalar_input else result

    def derivative(self, v_query):
        """分段常数导数 (分段线性的导数是分段常数).

        f'(v) = (f_{j+1} - f_j) / (v_{j+1} - v_j)  在区间 j 内
        """
        v_query = np.asarray(v_query, dtype=np.float64)
        scalar_input = v_query.ndim == 0
        v_query = np.atleast_1d(v_query)

        indices = np.searchsorted(self.v_grid, v_query, side='right') - 1
        indices = np.clip(indices, 0, self.N - 2)

        dv = self.v_grid[indices + 1] - self.v_grid[indices]
        dv = np.where(dv > 1e-30, dv, 1e-30)
        df = (self.f_values[indices + 1] - self.f_values[indices]) / dv

        return float(df[0]) if scalar_input else df


# ===========================================================================
#  §2  乘积求积规则  (源自 product_rule)
# ===========================================================================
def product_quadrature_3d_spherical(n_radial, rule_type="gauss-hermite"):
    """构造 3D 球坐标下的乘积求积规则.

    映射自 product_rule:
      原项目: 多维求积 = 1D 规则的张量积
              (x_{ijk}, w_{ijk}) = (x_i^x, w_i^x) ⊗ (x_j^y, w_j^y) ⊗ (x_k^z, w_k^z)
      本项目: 3D 速度空间的球面积分
              ∫_{R³} f(v) d³v = ∫₀^∞ ∫₀^π ∫₀^{2π} f(v,θ,φ) v² sinθ dφ dθ dv

    对于各向同性 f(v):
      ∫ f d³v = 4π ∫₀^∞ f(v) v² dv

    Parameters
    ----------
    n_radial : int  径向求积点数
    rule_type : str  "gauss-hermite" 或 "gauss-laguerre"

    Returns
    -------
    v_pts : ndarray  求积点 (径向)
    v_wts : ndarray  求积权重 (包含 v² 和球面积分因子)
    """
    if rule_type == "gauss-hermite":
        # Gauss-Hermite 求积: ∫_{-∞}^∞ exp(-x²) g(x) dx ≈ Σ w_i g(x_i)
        # 对于 ∫₀^∞ f(v) v² dv, 令 v² = t, dv = dt/(2√t)
        # ∫₀^∞ f(√t) t^{1/2} e^{-t} e^t dt
        # 使用 Gauss-Hermite 对称性: 只取正半轴
        x_gh, w_gh = np.polynomial.hermite.hermgauss(n_radial)
        # 只取正的节点
        pos = x_gh > 0
        v_pts = x_gh[pos]
        v_wts = 2.0 * w_gh[pos] * v_pts**2  # 包含 v² 因子
        # 加上球面因子 4π
        v_wts *= FOUR_PI
        return v_pts, v_wts

    elif rule_type == "gauss-laguerre":
        # Gauss-Laguerre: ∫₀^∞ e^{-x} g(x) dx ≈ Σ w_i g(x_i)
        x_gl, w_gl = np.polynomial.laguerre.laggauss(n_radial)
        v_pts = np.sqrt(x_gl)  # v = √x
        v_wts = 0.5 * w_gl * np.sqrt(x_gl) * FOUR_PI  # Jacobian + 球面
        return v_pts, v_wts

    elif rule_type == "trapezoidal":
        # 简单梯形法则
        v_pts = np.linspace(0.01, 6.0, n_radial)
        v_wts = np.full(n_radial, v_pts[1] - v_pts[0])
        v_wts[0] *= 0.5
        v_wts[-1] *= 0.5
        v_wts *= FOUR_PI * v_pts**2
        return v_pts, v_wts

    else:
        raise ValueError(f"未知求积规则类型: {rule_type}")


# ===========================================================================
#  §3  金字塔求积  (源自 pyramid_witherden_rule)
# ===========================================================================
def pyramid_velocity_time_quadrature(n_order, v_max, T_final):
    """速度-时间锥域上的求积规则.

    映射自 pyramid_witherden_rule:
      原项目: 方底金字塔 {(x,y,z): -1≤x,y≤1, 0≤z≤1} 上的求积
              精度阶数 0 ≤ p ≤ 10, 基于 Witherden & Vincent (2015)
      本项目: 速度-时间锥域 {(v,t): 0≤v≤v_max(1-t/T), 0≤t≤T}
              用于时空有限元框架中的 Fokker-Planck 积分.

    求积点分布:
      在时间方向: Gauss-Legendre 点
      在速度方向: 依赖于时间的自适应 Gauss 点

    Parameters
    ----------
    n_order : int  精度阶数 (0-10)
    v_max : float  最大速度
    T_final : float  最终时间

    Returns
    -------
    v_pts, t_pts : ndarray  求积点坐标
    weights : ndarray  求积权重
    """
    # 映射金字塔精度阶数到求积点数
    # (简化版 Witherden 规则)
    n_v = max(3, n_order + 2)
    n_t = max(2, n_order // 2 + 1)

    # 时间方向: Gauss-Legendre on [0, T]
    t_raw, w_t = np.polynomial.legendre.leggauss(n_t)
    t_pts_1d = 0.5 * T_final * (t_raw + 1.0)
    w_t *= 0.5 * T_final

    # 速度方向: Gauss-Legendre on [0, v_max] for each time
    v_raw, w_v = np.polynomial.legendre.leggauss(n_v)
    v_pts_1d = 0.5 * v_max * (v_raw + 1.0)
    w_v_base = 0.5 * v_max * w_v

    # 构造张量积
    all_v = []
    all_t = []
    all_w = []

    for j in range(n_t):
        t_j = t_pts_1d[j]
        # 在时间 t_j, 速度上限为 v_max * (1 - t_j/T)
        v_scale = max(0.01, 1.0 - t_j / T_final)
        for i in range(n_v):
            all_v.append(v_pts_1d[i] * v_scale)
            all_t.append(t_j)
            all_w.append(w_v_base[i] * v_scale * w_t[j])

    return np.array(all_v), np.array(all_t), np.array(all_w)


# ===========================================================================
#  §4  速度空间矩计算
# ===========================================================================
def compute_moments(x, f, n_quad=None):
    """计算分布函数的速度空间矩.

    物理矩:
      n = ∫ f d³v = 4π ∫ f(v) v² dv           (数密度)
      n⟨v²⟩ = 4π ∫ f(v) v⁴ dv                  (二阶矩 → 温度)
      n⟨v⁴⟩ = 4π ∫ f(v) v⁶ dv                  (四阶矩 → 热流修正)

    使用复合 Gauss-Legendre 求积.
    """
    # 被积函数
    integrand_n = FOUR_PI * x**2 * f
    integrand_T = FOUR_PI * x**4 * f
    integrand_q = FOUR_PI * x**6 * f

    # 梯形积分 (简单可靠)
    n_moment = np.trapz(integrand_n, x)
    T_moment = np.trapz(integrand_T, x)
    q_moment = np.trapz(integrand_q, x)

    # 导出物理量
    density = n_moment
    temperature = T_moment / (3.0 * n_moment) if n_moment > 1e-30 else 0.0
    # 热流 (三阶矩)
    heat_flux = q_moment / n_moment if n_moment > 1e-30 else 0.0

    return {
        "density": density,
        "temperature": temperature,
        "mean_v2": T_moment / n_moment if n_moment > 1e-30 else 0.0,
        "mean_v4": q_moment / n_moment if n_moment > 1e-30 else 0.0,
        "heat_flux": heat_flux,
    }


# ===========================================================================
#  §5  Rosenbluth 势积分
# ===========================================================================
def compute_rosenbluth_H(x, f):
    """Rosenbluth 势 H(v):

    H(v) = 4π [ (1/v) ∫₀ᵛ f(v') v'² dv' + ∫ᵛ^∞ f(v') v' dv' ]

    物理含义: H(v) 与动力学摩擦系数相关.
    A(v) = Γ ∂H/∂v

    数值实现: 两次累积积分
    """
    N = len(x)
    dx_arr = np.diff(x)

    # 内部积分: I₁(v) = ∫₀ᵛ f(v') v'² dv'
    integrand1 = f * x**2
    I1 = np.zeros(N)
    for i in range(1, N):
        I1[i] = I1[i-1] + 0.5 * (integrand1[i] + integrand1[i-1]) * dx_arr[i-1]
    I1 *= FOUR_PI

    # 外部积分: I₂(v) = ∫ᵛ^∞ f(v') v' dv'
    integrand2 = f * x
    I2_total = np.trapz(integrand2, x) * FOUR_PI
    I2_cumul = np.zeros(N)
    for i in range(1, N):
        I2_cumul[i] = I2_cumul[i-1] + 0.5 * (integrand2[i] + integrand2[i-1]) * dx_arr[i-1]
    I2_cumul *= FOUR_PI
    I2 = I2_total - I2_cumul

    # H(v)
    H = np.zeros(N)
    for i in range(N):
        if x[i] > 1e-10:
            H[i] = I1[i] / x[i] + I2[i]
        else:
            H[i] = I2[0]  # lim v→0: H(0) = ∫₀^∞ f v' dv'

    return H


def compute_rosenbluth_G_potential(x, f):
    """Rosenbluth 势 G(v):

    G(v) = (1/2) · 4π [ (1/v) ∫₀ᵛ f(v') v'⁴ dv'/3
                       + ∫ᵛ^∞ f(v') v'³ dv'/3 + v ∫₀ᵛ f(v') v' dv' ]

    简化: G(v) = (4π/(2v)) [∫₀ᵛ f v'⁴ dv'/3 + v² ∫₀ᵛ f v'² dv' + ...]

    G(v) 与扩散系数相关: D(v) = Γ/(2v) ∂²G/∂v²
    """
    N = len(x)
    dx_arr = np.diff(x)

    # ∫₀ᵛ f(v') v'⁴ dv'
    int_v4 = np.zeros(N)
    integrand_v4 = f * x**4
    for i in range(1, N):
        int_v4[i] = int_v4[i-1] + 0.5 * (integrand_v4[i] + integrand_v4[i-1]) * dx_arr[i-1]

    # ∫₀ᵛ f(v') v'² dv'
    int_v2 = np.zeros(N)
    integrand_v2 = f * x**2
    for i in range(1, N):
        int_v2[i] = int_v2[i-1] + 0.5 * (integrand_v2[i] + integrand_v2[i-1]) * dx_arr[i-1]

    G_pot = np.zeros(N)
    for i in range(N):
        if x[i] > 1e-10:
            G_pot[i] = (FOUR_PI / (2.0 * x[i])) * (
                int_v4[i] / 3.0 + x[i]**2 * int_v2[i]
            )
        else:
            G_pot[i] = 0.0

    return G_pot
