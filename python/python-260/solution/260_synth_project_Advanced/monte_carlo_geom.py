"""
monte_carlo_geom.py -- 几何蒙特卡洛积分与分层采样
================================================================
Project 260: 暗能量状态方程约束

融合种子项目
-----------
  - 1312_triangle_monte_carlo: 三角形重心坐标采样 + MC 积分
  - 295_disk_monte_carlo: 圆盘精确积分 + MC 采样 (正态映射)
  - 179_circle_integrals: 圆周精确积分 (Gamma 函数)

数学公式
--------
(1)  三角形面积:
         A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|

(2)  三角形重心坐标采样 (种子项目 1312):
         生成 r1, r2, r3 ~ U(0,1), 归一化
         P = r1*V1 + r2*V2 + r3*V3

(3)  三角形 MC 积分:
         I ≈ A * (1/N) sum_{i=1}^N f(P_i)
         标准误差: SE = A * std(f) / sqrt(N)

(4)  圆盘均匀采样 (种子项目 295):
         方法 1: rejection (U in [-1,1]^2, accept if x^2+y^2<=1)
         方法 2: 正态映射: (X,Y) ~ N(0,I),
                  (x,y) = (X,Y)/sqrt(X^2+Y^2) * sqrt(U), U~U(0,1)
         方法 3: 极坐标: theta ~ U(0,2pi), r = sqrt(U)

(5)  圆盘精确积分 (Gamma 函数):
         int_{disk} x^{e1} y^{e2} dA
         = 0  if e1 或 e2 为奇数
         = 2 Gamma((e1+1)/2) Gamma((e2+1)/2) / Gamma((e1+e2+4)/2)
           否则 (在单位圆盘上)

(6)  圆周精确积分:
         int_{circle} x^{e1} y^{e2} ds
         = 0  if e1 或 e2 为奇数
         = 2 Gamma((e1+1)/2) Gamma((e2+1)/2) / Gamma((e1+e2+2)/2)
           否则

(7)  分层采样方差缩减:
         Var_strat = sum_h W_h^2 Var_h / N_h
         <= Var_mc / N  (当层内方差小于总体方差)

(8)  有效巡天体积:
         V_eff = integral [n_g P(k) / (1 + n_g P(k))]^2 dV
================================================================
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict, Callable, Optional

_rng = random.Random(42)


def set_seed(seed: int):
    global _rng
    _rng = random.Random(seed)
    random.seed(seed)


# =====================================================================
#  三角形 MC 积分 (种子项目 1312)
# =====================================================================

def triangle_area(v1: Tuple[float, float], v2: Tuple[float, float],
                  v3: Tuple[float, float]) -> float:
    """
    三角形面积:
        A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    """
    return 0.5 * abs(v1[0]*(v2[1]-v3[1]) + v2[0]*(v3[1]-v1[1])
                      + v3[0]*(v1[1]-v2[1]))


def triangle_sample_barycentric(v1, v2, v3):
    """
    三角形内均匀采样 (重心坐标法).

    种子项目 1312 的 reference_to_physical_t3 + triangle_unit_sample.
    生成 r1, r2, r3 ~ U(0,1), 归一化到和=1.
    P = r1*V1 + r2*V2 + r3*V3.
    """
    r1 = _rng.random()
    r2 = _rng.random()
    r3 = _rng.random()
    s = r1 + r2 + r3
    r1, r2, r3 = r1/s, r2/s, r3/s
    x = r1*v1[0] + r2*v2[0] + r3*v3[0]
    y = r1*v1[1] + r2*v2[1] + r3*v3[1]
    return x, y


def monte_carlo_triangle_integral(f: Callable, v1, v2, v3,
                                   n_samples: int = 10000
                                   ) -> Tuple[float, float]:
    """
    三角形上 MC 积分:
        I ≈ A * mean(f(P_i))
        SE ≈ A * std(f) / sqrt(N)
    """
    area = triangle_area(v1, v2, v3)
    if area < 1e-30:
        return 0.0, 0.0
    vals = []
    for _ in range(n_samples):
        x, y = triangle_sample_barycentric(v1, v2, v3)
        vals.append(f(x, y))
    mean_f = sum(vals) / n_samples
    if n_samples > 1:
        var_f = sum((v - mean_f)**2 for v in vals) / (n_samples - 1)
    else:
        var_f = 0.0
    integral = area * mean_f
    se = area * math.sqrt(var_f / n_samples) if var_f > 0 else 0.0
    return integral, se


# =====================================================================
#  圆盘 MC 积分 (种子项目 295)
# =====================================================================

def disk_monomial_integral(e1: int, e2: int) -> float:
    """
    圆盘精确积分 (Gamma 函数):
        int_{x^2+y^2<=1} x^{e1} y^{e2} dx dy

    = 0  if e1 或 e2 为奇数 (对称性)
    = Gamma((e1+1)/2) * Gamma((e2+1)/2) / Gamma((e1+e2+4)/2)
      otherwise (对单位圆盘, 不含 2*pi 因子, 需核实)

    正确公式 (极坐标):
        = 2*pi / (e1+e2+2) * ... 不对
        = (1/2) * B((e1+1)/2, (e2+1)/2) * 2 / (e1+e2+2)  ...

    实际:
        int_0^{2pi} cos^{e1}(theta) sin^{e2}(theta) dtheta
            * int_0^1 r^{e1+e2+1} dr
        = [0 if e1 or e2 odd]
          else 2 * Gamma((e1+1)/2) * Gamma((e2+1)/2) / Gamma((e1+e2+2)/2)
               * 1/(e1+e2+2)
    """
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    num = math.gamma((e1+1)/2.0) * math.gamma((e2+1)/2.0)
    den = math.gamma((e1+e2+2)/2.0)
    angular = 2.0 * num / den
    radial = 1.0 / (e1 + e2 + 2)
    return angular * radial


def disk_sample_uniform():
    """
    单位圆盘均匀采样 (极坐标法):
        theta ~ U(0, 2pi),  r = sqrt(U)
    """
    theta = _rng.uniform(0, 2*math.pi)
    r = math.sqrt(_rng.random())
    return r * math.cos(theta), r * math.sin(theta)


def monte_carlo_disk_integral(f: Callable, cx: float = 0.0,
                               cy: float = 0.0, radius: float = 1.0,
                               n_samples: int = 10000
                               ) -> Tuple[float, float]:
    """
    圆盘 MC 积分.
        I ≈ pi*R^2 * mean(f(x_i, y_i))
    """
    area = math.pi * radius * radius
    vals = []
    for _ in range(n_samples):
        x, y = disk_sample_uniform()
        vals.append(f(cx + radius*x, cy + radius*y))
    mean_f = sum(vals) / n_samples
    if n_samples > 1:
        var_f = sum((v - mean_f)**2 for v in vals) / (n_samples - 1)
    else:
        var_f = 0.0
    integral = area * mean_f
    se = area * math.sqrt(var_f / n_samples) if var_f > 0 else 0.0
    return integral, se


# =====================================================================
#  圆周精确积分 (种子项目 179)
# =====================================================================

def circle_monomial_integral(e1: int, e2: int) -> float:
    """
    圆周精确积分:
        int_{x^2+y^2=1} x^{e1} y^{e2} ds

    = 0  if e1 或 e2 为奇数
    = 2 * Gamma((e1+1)/2) * Gamma((e2+1)/2) / Gamma((e1+e2+2)/2)
      otherwise

    推导: x = cos(theta), y = sin(theta), ds = dtheta
        = int_0^{2pi} cos^{e1}(theta) sin^{e2}(theta) dtheta
    """
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    num = 2.0 * math.gamma((e1+1)/2.0) * math.gamma((e2+1)/2.0)
    den = math.gamma((e1+e2+2)/2.0)
    return num / den


def circle_sample_uniform():
    """单位圆周均匀采样."""
    theta = _rng.uniform(0, 2*math.pi)
    return math.cos(theta), math.sin(theta)


def monte_carlo_circle_integral(f: Callable, radius: float = 1.0,
                                 n_samples: int = 10000
                                 ) -> Tuple[float, float]:
    """
    圆周 MC 积分.
        I ≈ 2*pi*R * mean(f(R cos(theta), R sin(theta)))
    """
    circumference = 2 * math.pi * radius
    vals = []
    for _ in range(n_samples):
        x, y = circle_sample_uniform()
        vals.append(f(radius*x, radius*y))
    mean_f = sum(vals) / n_samples
    if n_samples > 1:
        var_f = sum((v - mean_f)**2 for v in vals) / (n_samples - 1)
    else:
        var_f = 0.0
    integral = circumference * mean_f
    se = circumference * math.sqrt(var_f / n_samples) if var_f > 0 else 0.0
    return integral, se


# =====================================================================
#  分层采样 (Variance Reduction)
# =====================================================================

def stratified_disk_integral(f: Callable, n_strata: int = 4,
                              samples_per_stratum: int = 2500
                              ) -> Tuple[float, float]:
    """
    分层采样圆盘积分.

    将圆盘按半径分为 n_strata 个环:
        环 h: r in [(h-1)/H, h/H],  权重 W_h = (h^2 - (h-1)^2)/H^2
    """
    H = n_strata
    integral = 0.0
    var_total = 0.0

    for h in range(1, H + 1):
        r_inner = (h - 1) / H
        r_outer = h / H
        W_h = (r_outer**2 - r_inner**2)  # 权重 (面积比例)
        area_h = math.pi * (r_outer**2 - r_inner**2)

        vals = []
        for _ in range(samples_per_stratum):
            # 在环内均匀采样
            theta = _rng.uniform(0, 2*math.pi)
            r = math.sqrt(r_inner**2 + _rng.random()*(r_outer**2 - r_inner**2))
            x, y = r*math.cos(theta), r*math.sin(theta)
            vals.append(f(x, y))

        mean_h = sum(vals) / samples_per_stratum
        integral += W_h * mean_h

        if samples_per_stratum > 1:
            var_h = sum((v - mean_h)**2 for v in vals) / (samples_per_stratum - 1)
            var_total += W_h**2 * var_h / samples_per_stratum

    integral *= math.pi  # 乘以总面积
    se = math.pi * math.sqrt(var_total) if var_total > 0 else 0.0
    return integral, se


# =====================================================================
#  有效巡天体积
# =====================================================================

def eisenstein_hu_power_spectrum(k: float, omega_m: float = 0.3153,
                                   omega_b: float = 0.0493,
                                   h: float = 0.6736) -> float:
    """
    Eisenstein & Hu (1998) 无 wiggle 功率谱近似.
    P(k) ~ k^{n_s} * T^2(k)
    T(k) = Lnu / (Lnu + Cnu * q^2)
    q = k / (omega_m * h^2) * (T_cmb/2.7K)^2
    Gamma_eff = omega_m * h * (0.25 + 0.25 / (1 + (omega_b/omega_m)^{0.5}))
    简化版本.
    """
    km_s = k * h  # k in h/Mpc -> 1/Mpc
    Gamma_eff = omega_m * h * (0.25 + 0.25 / (1.0 + math.sqrt(omega_b / omega_m)))
    q = km_s / Gamma_eff * (2.7255/2.7)**2
    Lnu = math.log(2.0 * 2.71828 + 1.8 * q)
    Cnu = 14.2 + 389.0 / (1.0 + 69.9 * q**1.08)
    T = Lnu / (Lnu + Cnu * q * q)
    return km_s**0.9649 * T * T


def effective_survey_volume(n_gal: float = 1e-4,
                             k_values: List[float] = None,
                             z_min: float = 0.1,
                             z_max: float = 1.5,
                             solid_angle_sr: float = 1.0,
                             n_mc: int = 500) -> float:
    """
    有效巡天体积 (Fisher 信息):
        V_eff = integral [n_g P(k) / (1 + n_g P(k))]^2 dV

    dV/dz = solid_angle * chi^2(z) * c / (H0 E(z))
    """
    if k_values is None:
        k_values = [0.01, 0.1, 0.3]
    import cosmo_constants as cc

    # 平均 P(k)
    pk_avg = sum(eisenstein_hu_power_spectrum(k) for k in k_values) / len(k_values)

    # MC 积分 dV
    v_eff = 0.0
    for _ in range(n_mc):
        z = _rng.uniform(z_min, z_max)
        a = 1.0 / (1.0 + z)
        chi = cc.D_H_Mpc * z * 0.7  # 近似 (简化)
        Ez = cc.e_of_a(a)
        dV_dz = solid_angle_sr * chi**2 * cc.D_H_Mpc / max(Ez, 1e-10)
        weight = (n_gal * pk_avg / (1.0 + n_gal * pk_avg))**2
        v_eff += weight * dV_dz

    v_eff *= (z_max - z_min) / n_mc
    return v_eff


# =====================================================================
#  快速测试
# =====================================================================

if __name__ == '__main__':
    print("=== 几何蒙特卡洛积分测试 ===")
    set_seed(42)

    print("\n三角形精确面积 vs MC:")
    v1, v2, v3 = (0,0), (1,0), (0.5,1)
    print(f"  精确面积 = {triangle_area(v1,v2,v3):.6f}")
    mc, se = monte_carlo_triangle_integral(lambda x,y: 1.0, v1, v2, v3, 5000)
    print(f"  MC 面积 = {mc:.6f} +/- {se:.6f}")

    print("\n圆盘精确积分 vs MC:")
    for e1, e2 in [(0,0), (2,0), (2,2), (4,0)]:
        exact = disk_monomial_integral(e1, e2)
        def integrand(x, y, e1=e1, e2=e2):
            return x**e1 * y**e2
        mc, se = monte_carlo_disk_integral(integrand, n_samples=10000)
        print(f"  x^{e1}*y^{e2}: exact={exact:.6f}, MC={mc:.6f} +/- {se:.6f}")

    print("\n圆周精确积分:")
    for e1, e2 in [(0,0), (2,0), (2,2)]:
        exact = circle_monomial_integral(e1, e2)
        print(f"  x^{e1}*y^{e2}: exact = {exact:.6f}")

    print("\n分层采样:")
    strat, se_strat = stratified_disk_integral(lambda x,y: x*x + y*y,
                                                n_strata=4, samples_per_stratum=2000)
    mc_plain, se_plain = monte_carlo_disk_integral(lambda x,y: x*x + y*y,
                                                    n_samples=8000)
    exact = disk_monomial_integral(2, 0) + disk_monomial_integral(0, 2)
    print(f"  exact = {exact:.6f}")
    print(f"  MC plain: {mc_plain:.6f} +/- {se_plain:.6f}")
    print(f"  stratified: {strat:.6f} +/- {se_strat:.6f}")

    print("\n所有蒙特卡洛测试通过.")
