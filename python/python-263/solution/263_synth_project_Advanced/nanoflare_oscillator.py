# -*- coding: utf-8 -*-
"""
nanoflare_oscillator.py
-----------------------
日冕纳耀斑 (nanoflare) 弛豫振荡器.

物理背景
--------
Parker (1988) 提出日冕加热由大量微小磁重联事件 ("纳耀斑") 驱动.
每个纳耀斑释放 ~1e24 J 能量, 持续时间约数十秒. 观测上 (Krucker & Benz
1998), 纳耀斑呈现"能量积累 - 快速释放 - 冷却恢复"的弛豫振荡特征.

模型
----
借鉴 FitzHugh-Nagumo / 心跳模型 (511_heartbeat_ode) 的弛豫振荡框架,
建立磁能 - 等离子体热能耦合的二维自治系统:

    dE_mag/dt = - (1/epsilon) * (E_mag^3 - a E_mag + W)  + S_ext
    dW/dt     = E_mag - gamma W

其中:
- E_mag 为无量纲磁自由能 (重联电流片能量)
- W 为等离子体内能辅助变量
- epsilon << 1 为快-慢时间尺度分离参数
- a 控制双稳阈值 (典型 0.7-1.0)
- gamma 表示热损失率 (辐射 + 热传导)
- S_ext 为外部驱动 (光球足点剪切流注入 Poynting 通量)

临界条件
--------
当 E_mag > E_thresh = sqrt(a/3), 系统越过鞍点进入快速释放相.
一个完整周期的特征时间:
    T_cycle ~ (2 pi / sqrt(a)) * epsilon + tau_cool
其中 tau_cool 为辐射冷却时标.

参数典型值 (日冕)
-----------------
- epsilon ~ 0.05 (磁积累时标/释放时标)
- a ~ 0.85
- gamma ~ 0.1 (无量纲)
- S_ext ~ 0.02 (归一化 Poynting 注入)
"""
from __future__ import annotations
import numpy as np


def nanoflare_parameters() -> dict:
    """默认纳耀斑振荡参数."""
    return dict(
        epsilon=0.05,
        a=0.85,
        gamma=0.10,
        s_ext=0.02,
        t_stop=300.0,      # 无量纲时间
        dt=0.05,
        e_thresh=np.sqrt(0.85 / 3.0),
    )


def nanoflare_rhs(t: float, y: np.ndarray,
                  epsilon: float = 0.05, a: float = 0.85,
                  gamma: float = 0.10, s_ext: float = 0.02) -> np.ndarray:
    """二维自治系统右端.

    y[0] = E_mag, y[1] = W
    """
    e_mag, w = y[0], y[1]
    # 正则化立方项防止溢出
    e_cubed = e_mag * (e_mag**2 + 1.0e-12)
    de = -(1.0 / epsilon) * (e_cubed - a * e_mag + w) + s_ext
    dw = e_mag - gamma * w
    return np.array([de, dw])


def integrate_nanoflare(y0: np.ndarray, t_span, dt: float,
                        params: dict | None = None):
    """前向 Euler + 自适应子步 (显式 RK2) 积分振荡器."""
    if params is None:
        params = nanoflare_parameters()
    eps = params["epsilon"]
    a = params["a"]
    gam = params["gamma"]
    s = params["s_ext"]

    t0, t1 = t_span
    n_step = max(2, int(np.ceil((t1 - t0) / dt)))
    dt_loc = (t1 - t0) / n_step
    t_vals = np.linspace(t0, t1, n_step + 1)
    y_vals = np.zeros((n_step + 1, 2))
    y_vals[0] = y0

    for k in range(n_step):
        y = y_vals[k]
        t = t_vals[k]
        # RK2 (midpoint)
        def f(yy):
            return nanoflare_rhs(t, yy, eps, a, gam, s)
        k1 = f(y)
        k2 = f(y + 0.5 * dt_loc * k1)
        y_new = y + dt_loc * k2
        # 物理约束: E_mag >= 0
        y_new[0] = max(y_new[0], 0.0)
        y_vals[k + 1] = y_new

    return t_vals, y_vals


def nanoflare_energy_release_rate(y: np.ndarray,
                                  epsilon: float = 0.05) -> float:
    """瞬时能量释放率 (正值为释放, 负值为积累):

    dQ/dt = (1/epsilon) * (E^3 - a E + W)
    """
    return (1.0 / epsilon) * (y[0]**3 - 0.85 * y[0] + y[1])


def detect_nanoflare_events(t_vals: np.ndarray,
                            e_mag_vals: np.ndarray,
                            threshold: float) -> list:
    """检测纳耀斑事件: 当 dE_mag/dt < -threshold 时触发.

    返回每个事件的 (t_start, t_peak, t_end, delta_E).
    """
    events = []
    dE = np.gradient(e_mag_vals, t_vals)
    in_event = False
    t_start = 0.0
    e_start = 0.0
    peak_drop = 0.0
    for k in range(1, len(t_vals)):
        rate = -dE[k]
        if rate > threshold and not in_event:
            in_event = True
            t_start = t_vals[k]
            e_start = e_mag_vals[k]
            peak_drop = rate
        elif in_event:
            peak_drop = max(peak_drop, rate)
            if rate < threshold * 0.1:
                local_idx = np.argmin(e_mag_vals[max(0, k - 10):k + 1])
                t_peak_idx = max(0, k - 10) + local_idx
                events.append(dict(
                    t_start=float(t_start),
                    t_end=float(t_vals[k]),
                    t_peak=float(t_vals[t_peak_idx]),
                    delta_e=float(e_start - e_mag_vals[k]),
                    peak_rate=float(peak_drop),
                ))
                in_event = False
    return events


def power_law_index(events: list, bin_num: int = 10) -> float:
    """拟合纳耀斑能量分布 dN/dE \propto E^{-alpha}.

    alpha > 2 意味着总加热由最小事件主导 (Parker 1988).
    """
    if len(events) < 3:
        return 2.0
    de = np.array([ev["delta_e"] for ev in events if ev["delta_e"] > 1.0e-6])
    if de.size < 3:
        return 2.0
    hist, edges = np.histogram(np.log10(de), bins=bin_num)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    if mask.sum() < 2:
        return 2.0
    log_n = np.log10(hist[mask].astype(float))
    log_e = centers[mask]
    # 最小二乘 fit log N = -alpha * log E + const
    A = np.vstack([log_e, np.ones_like(log_e)]).T
    coef, _, _, _ = np.linalg.lstsq(A, log_n, rcond=None)
    return float(-coef[0])
