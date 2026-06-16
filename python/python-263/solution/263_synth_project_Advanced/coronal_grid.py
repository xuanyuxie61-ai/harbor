# -*- coding: utf-8 -*-
"""
coronal_grid.py
---------------
日冕-太阳风自适应网格生成: 基于重心 Voronoi 剖分 (CVT) 的采样-Lloyd 算法.

物理动机
--------
在日冕加热的"热斑"模型中, 加热率 Q(r) 沿环轴呈现高度非均匀分布
(例如在环顶处尖锐峰值, 在环足部陡降). 为在有限自由度下最大限度地
降低空间离散误差, 我们采用 CVT 自适应网格, 使节点密度正比于
|d^2 T/ds^2|^{1/3} (等分布原理, de Boor 1973):

    rho(s) \propto |T''(s)|^{1/3}

对太阳风加速区, 则使用 Parker 跨声速过渡的梯度:

    rho(s) \propto |d ln M / ds|^{1/2},  M = v/c_s

算法
----
1) 在参考区间 [R_sun, R_out] 生成 N 个初始发生器 z_i.
2) 每次迭代: 以 rho(s) 为密度, 采样 S 个点 {x_k}.
3) 将每个 x_k 归属到最近的 z_i (Voronoi region).
4) 更新 z_i = 区域重心 (centroid):
       z_i <- sum_{k in V_i} rho(x_k) x_k / sum_{k in V_i} rho(x_k)
5) 收敛判据: max_i |z_i^{(k+1)} - z_i^{(k)}| < tol.

参考文献
--------
- Du, Faber & Gunzburger (1999), "Centroidal Voronoi Tessellations"
- Burkardt (2010), cvt_1d_sampling (MATLAB 原版)
- Maxson et al. (2017), "Adaptive mesh for solar wind"
"""
from __future__ import annotations
import numpy as np
from solar_constants import SOLAR_RADIUS

# ============================================================
# 日冕环轴方向上的加热率密度 (用作 CVT 权重)
# ============================================================
def coronal_heating_profile(s: np.ndarray, loop_half_length: float,
                            s_footpoint: float) -> np.ndarray:
    """双足加热 + 环顶峰值模型 (Rosner-Tucker-Vaiana 1978 推广):

    Q(s) = Q_0 * [ exp(- (s - s_fp)^2 / lambda_fp^2)
                 + exp(- (s + s_fp)^2 / lambda_fp^2) ]
         + Q_apex * exp(- (s - 0)^2 / lambda_apex^2)

    其中 s=0 为环顶, s = \pm s_fp 为足点.
    """
    lambda_fp = 0.08 * loop_half_length
    lambda_apex = 0.15 * loop_half_length
    q_fp = 1.0
    q_apex = 0.5
    term_fp = (
        np.exp(-((s - s_footpoint) ** 2) / lambda_fp**2)
        + np.exp(-((s + s_footpoint) ** 2) / lambda_fp**2)
    )
    term_apex = np.exp(-(s ** 2) / lambda_apex**2)
    return q_fp * term_fp + q_apex * term_apex


def solar_wind_density_weight(r: np.ndarray, r_c: float) -> np.ndarray:
    """太阳风跨声速过渡的网格密度权重:

    rho(r) \propto 1 + A * exp( - (r - r_c)^2 / sigma^2 )

    使网格在临界点 r_c 附近加密.
    """
    sigma = 0.3 * r_c
    return 1.0 + 8.0 * np.exp(-((r - r_c) ** 2) / sigma**2)


# ============================================================
# CVT 采样-Lloyd 算法
# ============================================================
def cvt_lloyd_sampling(r_inner: float, r_outer: float,
                       n_gen: int, it_num: int,
                       s_num: int, rho_func,
                       seed: int = 263, tol: float = 1.0e-6):
    """CVT 采样-Lloyd 迭代.

    Parameters
    ----------
    r_inner, r_outer : float
        积分区间 [R_sun, R_out].
    n_gen : int
        发生器 (网格节点) 个数.
    it_num : int
        Lloyd 迭代最大次数.
    s_num : int
        每轮采样点数 (用于估计 Voronoi 区域).
    rho_func : callable
        密度函数 rho(r) >= 0.
    seed : int
        随机种子 (保证可复现).
    tol : float
        位移收敛阈值.

    Returns
    -------
    z : ndarray (n_gen,)
        收敛后的发生器位置 (升序).
    history : list of float
        每轮最大位移.
    """
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(r_inner, r_outer, size=n_gen))
    history: list[float] = []

    for it in range(it_num):
        samples = rng.uniform(r_inner, r_outer, size=s_num)
        weights = rho_func(samples)
        # 对每个采样点归属最近发生器
        # 使用搜索排序以 O(s_num log n_gen) 替代 O(s_num * n_gen)
        idx = np.searchsorted(z, samples, side="right") - 1
        idx = np.clip(idx, 0, n_gen - 2)
        # 比较与 z[idx] 和 z[idx+1] 的距离
        left_d = np.abs(samples - z[idx])
        right_d = np.abs(samples - z[idx + 1])
        assign = np.where(left_d <= right_d, idx, idx + 1)

        z_new = np.empty_like(z)
        max_shift = 0.0
        for i in range(n_gen):
            mask = assign == i
            if np.any(mask):
                w_sum = weights[mask].sum()
                if w_sum > 1.0e-30:
                    z_new[i] = (weights[mask] * samples[mask]).sum() / w_sum
                else:
                    z_new[i] = z[i]
            else:
                # 孤立发生器: 与左右邻居取中点
                if i == 0:
                    z_new[i] = 0.5 * (r_inner + z[1])
                elif i == n_gen - 1:
                    z_new[i] = 0.5 * (z[-2] + r_outer)
                else:
                    z_new[i] = 0.5 * (z[i - 1] + z[i + 1])
            max_shift = max(max_shift, abs(z_new[i] - z[i]))
        z = np.sort(z_new)
        history.append(max_shift)
        if max_shift < tol:
            break
    return z, history


# ============================================================
# 非均匀网格上的差分算子辅助
# ============================================================
def nonuniform_dx(z: np.ndarray) -> np.ndarray:
    """返回相邻节点间距 h_i = z_{i+1} - z_i, 长度 n-1."""
    return z[1:] - z[:-1]


def harmonic_average(h: np.ndarray) -> np.ndarray:
    """相邻单元面调和平均, 长度 n-2:
    h_{i+1/2} = 2 h_i h_{i+1} / (h_i + h_{i+1})."""
    return 2.0 * h[:-1] * h[1:] / (h[:-1] + h[1:] + 1.0e-30)


def grid_quality(z: np.ndarray) -> dict:
    """网格质量诊断: 最大/最小/平均比、最大位移偏离."""
    h = nonuniform_dx(z)
    ratio = h.max() / (h.min() + 1.0e-30)
    return dict(
        n_nodes=int(z.size),
        h_min=float(h.min()),
        h_max=float(h.max()),
        h_mean=float(h.mean()),
        ratio=float(ratio),
    )


# ============================================================
# 便捷入口: 生成日冕环网格 / 太阳风网格
# ============================================================
def build_corona_loop_grid(s_fp: float = 1.0e8, n_gen: int = 64,
                          it_num: int = 80, s_num: int = 5000,
                          seed: int = 263) -> np.ndarray:
    """构建 [-s_fp, s_fp] 上的日冕环自适应网格."""
    def rho(s):
        return coronal_heating_profile(s, s_fp, s_fp) + 0.1
    z, _ = cvt_lloyd_sampling(-s_fp, s_fp, n_gen, it_num, s_num, rho, seed)
    return z


def build_solar_wind_grid(r_out: float = 20.0 * SOLAR_RADIUS,
                          n_gen: int = 80, it_num: int = 80,
                          s_num: int = 6000,
                          t_corona: float = 1.5e6,
                          seed: int = 263) -> np.ndarray:
    """构建 [R_sun, R_out] 上的太阳风径向自适应网格."""
    from solar_constants import parker_critical_radius, sound_speed
    cs = sound_speed(t_corona)
    r_c = parker_critical_radius(t_corona)

    def rho(r):
        return solar_wind_density_weight(r, r_c)
    z, _ = cvt_lloyd_sampling(SOLAR_RADIUS, r_out, n_gen, it_num,
                              s_num, rho, seed)
    return z


if __name__ == "__main__":
    z = build_corona_loop_grid()
    print("Corona loop grid:", z[:5], "...", z[-5:])
    print("Quality:", grid_quality(z))
    z2 = build_solar_wind_grid()
    print("Solar wind grid:", z2[:5], "...", z2[-5:])
    print("Quality:", grid_quality(z2))
