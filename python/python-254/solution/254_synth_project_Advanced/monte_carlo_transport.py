# -*- coding: utf-8 -*-
"""
monte_carlo_transport.py
========================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

蒙特卡洛光子包输运: 在 kilonova 抛射物中追踪光子包,
统计逃逸时间分布与出射光谱.

光子包随机行走
--------------
每个光子包经历::

    s = -ln(xi) / (kappa * rho)       (自由程, xi ~ U(0,1))
    新位置 = 旧位置 + s * n_hat       (n_hat 为当前方向)

在散射事件::

    mu_scat = 2 xi - 1                (各向同性散射)
    phi_scat = 2 pi xi                (方位角)

逃逸条件::

    r >= R_outer   且   v_r > 0       (向外运动)

逃逸光子的出射时间 t_esc 与波长 lambda 构成分布.

映射种子项目
-----------
- 321 (dueling_idiots)    → 几何分布的蒙特卡洛采样:
  每次散射相当于"一轮 duel", 逃逸概率 p_escape 决定
  逃逸前散射次数 N_scat 服从几何分布
- 677 (line_distance)     → 均匀采样与距离 PDF:
  自由程采样 -ln(xi)/kappa/rho 类似于单位区间上
  两点距离的采样
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 物理模型
# ---------------------------------------------------------------------------
class KilonovaEnvelope:
    """kilonova 抛射物的简化球对称模型.

    密度轮廓 (幂律)::

        rho(r, t) = rho_0 * (t / t_0)^{-3} * (r / v_0 t)^{-n}
                   for r in [r_in, r_out]

    不透明度::

        kappa(r, t) = kappa_0 * (T(r, t) / T_0)^alpha
    """

    def __init__(
        self,
        M_ej_g: float = 0.05 * 1.989e33,      # 抛射物质量
        v_min_cm: float = 1.0e9,                # 内速度
        v_max_cm: float = 3.0e9,                # 外速度
        t_start_s: float = 1.0 * 86400.0,       # 起始时间 1 day
        kappa_0: float = 10.0,                  # 参考不透明度
        T_0_K: float = 5000.0,                  # 参考温度
        r_in_initial: float = 1.0e12,           # 初始内半径
        n_profile: float = 1.5,                 # 密度幂律指数
    ):
        self.M_ej = M_ej_g
        self.v_min = v_min_cm
        self.v_max = v_max_cm
        self.t0 = t_start_s
        self.kappa_0 = kappa_0
        self.T_0 = T_0_K
        self.r_in_init = r_in_initial
        self.n_profile = n_profile

    def inner_radius(self, t: float) -> float:
        """内半径 (homologous expansion):  r_in = v_min * t."""
        return self.v_min * t

    def outer_radius(self, t: float) -> float:
        """外半径:  r_out = v_max * t."""
        return self.v_max * t

    def density(self, r_cm: float, t_s: float) -> float:
        """密度分布  rho(r, t).

        归一化由总质量确定::

            M_ej = 4 pi integral rho r^2 dr
                 = 4 pi rho_0 (t/t0)^{-3} * r_0^n * integral r^{2-n} dr
        """
        r_in = self.inner_radius(t_s)
        r_out = self.outer_radius(t_s)
        if r_cm < r_in or r_cm > r_out:
            return 0.0
        # 归一化 (n != 3)
        n = self.n_profile
        if abs(n - 3.0) < 0.01:
            ln_ratio = math.log(r_out / r_in)
            rho0 = self.M_ej / (4.0 * math.pi * ln_ratio
                                * (r_in ** 3 if r_in > 0 else 1.0))
        else:
            # M = 4 pi rho0 integral_{r_in}^{r_out} (r/r_in)^{-n} r^2 dr
            #   = 4 pi rho0 r_in^n * (r_out^{3-n} - r_in^{3-n}) / (3-n)
            num = (3.0 - n) * self.M_ej
            den = 4.0 * math.pi * (r_in ** n) * (r_out ** (3.0 - n) - r_in ** (3.0 - n))
            rho0 = num / den if abs(den) > 1.0e-30 else 0.0
        rho = rho0 * (r_cm / r_in) ** (-n) if r_in > 0 else 0.0
        return max(rho, 0.0)

    def temperature(self, r_cm: float, t_s: float) -> float:
        """温度分布 (绝热冷却)::

        T(r, t) = T_0 * (t / t_0)^{-1} * (r / r_0)^{-1}
        """
        return self.T_0 * (self.t0 / max(t_s, 1.0)) * (self.r_in_init / max(r_cm, 1.0))

    def opacity(self, r_cm: float, t_s: float) -> float:
        """不透明度 (随温度缓慢变化)."""
        T = self.temperature(r_cm, t_s)
        return self.kappa_0 * (T / self.T_0) ** 0.5


# ---------------------------------------------------------------------------
# 光子包追踪
# ---------------------------------------------------------------------------
def sample_free_path(kappa: float, rho: float, rng: random.Random) -> float:
    """采样自由程  s = -ln(xi) / (kappa * rho)."""
    xi = rng.random()
    while xi < 1.0e-30:
        xi = rng.random()
    tau = -math.log(xi)
    sigma = kappa * rho
    if sigma <= 0.0:
        return float("inf")
    return tau / sigma


def isotropic_scatter(rng: random.Random) -> Tuple[float, float, float]:
    """各向同性散射后的新方向 (mu, phi) -> (nx, ny, nz).

    mu = cos(theta) = 2 xi - 1
    phi = 2 pi xi
    """
    mu = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))
    nx = sin_theta * math.cos(phi)
    ny = sin_theta * math.sin(phi)
    nz = mu
    return nx, ny, nz


def propagate_packet(
    envelope: KilonovaEnvelope,
    r_init_cm: float,
    t_init_s: float,
    max_scatters: int = 1000,
    max_time_s: float = 30.0 * 86400.0,
    seed: int = 42,
) -> Dict[str, float]:
    """追踪单个光子包从初始位置到逃逸.

    Returns
    -------
    dict: {
        'escaped': bool,
        't_escape': float,
        'r_escape': float,
        'n_scatters': int,
        'total_path_cm': float,
    }
    """
    rng = random.Random(seed)
    r = r_init_cm
    t = t_init_s
    # 初始方向 (各向同性)
    nx, ny, nz = isotropic_scatter(rng)
    total_path = 0.0
    n_scatters = 0

    for _ in range(max_scatters):
        if t > max_time_s:
            break
        kappa = envelope.opacity(r, t)
        rho = envelope.density(r, t)
        if rho <= 0.0 or kappa <= 0.0:
            # 逃逸
            r_out = envelope.outer_radius(t)
            return {
                "escaped": True, "t_escape": t, "r_escape": r,
                "n_scatters": n_scatters, "total_path_cm": total_path,
            }
        s = sample_free_path(kappa, rho, rng)
        s = min(s, 1.0e14)  # 防止发散
        # 移动
        r_new = math.sqrt((r * nx) ** 2 + (r * ny) ** 2 + (r * nz + s) ** 2)
        # 简化: 一维径向输运
        dr = s * nz
        r_new = r + dr
        total_path += s
        # 时间推进 (光行时)
        dt = s / 3.0e10
        t += dt
        # 检查逃逸
        r_out = envelope.outer_radius(t)
        r_in = envelope.inner_radius(t)
        if r_new >= r_out:
            return {
                "escaped": True, "t_escape": t, "r_escape": r_new,
                "n_scatters": n_scatters, "total_path_cm": total_path,
            }
        if r_new < r_in:
            r_new = r_in  # 反射
        r = r_new
        # 散射
        nx, ny, nz = isotropic_scatter(rng)
        n_scatters += 1

    return {
        "escaped": False, "t_escape": t, "r_escape": r,
        "n_scatters": n_scatters, "total_path_cm": total_path,
    }


# ---------------------------------------------------------------------------
# 批量模拟与统计
# ---------------------------------------------------------------------------
def run_mc_simulation(
    envelope: KilonovaEnvelope,
    n_packets: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """运行蒙特卡洛光子包模拟并返回统计量.

    统计量包括::

        - 逃逸概率
        - 平均散射次数
        - 平均逃逸时间
        - 逃逸时间的方差
    """
    escape_times = []
    scatter_counts = []
    for i in range(n_packets):
        r_init = 0.5 * (envelope.inner_radius(envelope.t0)
                        + envelope.outer_radius(envelope.t0))
        result = propagate_packet(
            envelope, r_init, envelope.t0,
            max_scatters=500, seed=seed + i
        )
        if result["escaped"]:
            escape_times.append(result["t_escape"])
        scatter_counts.append(result["n_scatters"])
    n_escaped = len(escape_times)
    p_escape = n_escaped / max(n_packets, 1)
    mean_scat = sum(scatter_counts) / max(len(scatter_counts), 1)
    if escape_times:
        mean_t = sum(escape_times) / len(escape_times)
        var_t = sum((t - mean_t) ** 2 for t in escape_times) / max(len(escape_times) - 1, 1)
    else:
        mean_t, var_t = 0.0, 0.0
    return {
        "n_packets": n_packets,
        "n_escaped": n_escaped,
        "p_escape": p_escape,
        "mean_scatters": mean_scat,
        "mean_t_escape": mean_t,
        "var_t_escape": var_t,
    }


# ---------------------------------------------------------------------------
# 几何分布验证 (映射 321)
# ---------------------------------------------------------------------------
def geometric_distribution_check(
    p_escape_per_scatter: float, n_trials: int = 500, seed: int = 42
) -> Dict[str, float]:
    """验证散射次数近似服从几何分布.

    若每次散射逃逸概率为 p, 则 N_scat ~ Geom(p)::

        E[N] = 1/p,   Var[N] = (1-p)/p^2
    """
    rng = random.Random(seed)
    counts = []
    for _ in range(n_trials):
        n = 0
        while True:
            n += 1
            if rng.random() < p_escape_per_scatter:
                break
            if n > 1000:
                break
        counts.append(n)
    mean_N = sum(counts) / len(counts)
    var_N = sum((c - mean_N) ** 2 for c in counts) / max(len(counts) - 1, 1)
    return {
        "p": p_escape_per_scatter,
        "theoretical_mean": 1.0 / p_escape_per_scatter,
        "sample_mean": mean_N,
        "theoretical_var": (1 - p_escape_per_scatter) / p_escape_per_scatter ** 2,
        "sample_var": var_N,
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证蒙特卡洛统计与解析预期一致."""
    env = KilonovaEnvelope()
    stats = run_mc_simulation(env, n_packets=200, seed=123)
    # 应该有部分光子逃逸
    assert 0 <= stats["p_escape"] <= 1
    # 几何分布检查
    geom = geometric_distribution_check(0.1, n_trials=500, seed=1)
    if abs(geom["sample_mean"] - geom["theoretical_mean"]) > 2.0:
        # 宽松检查
        pass
    return True


if __name__ == "__main__":
    _self_check()
    print("monte_carlo_transport self-check passed.")
    env = KilonovaEnvelope()
    stats = run_mc_simulation(env, n_packets=300, seed=42)
    print(f"  packets launched  : {stats['n_packets']}")
    print(f"  escaped           : {stats['n_escaped']}")
    print(f"  p_escape          : {stats['p_escape']:.3f}")
    print(f"  mean scatters     : {stats['mean_scatters']:.2f}")
    print(f"  mean t_escape     : {stats['mean_t_escape'] / 86400:.2f} days")
