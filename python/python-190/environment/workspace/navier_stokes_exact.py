"""
navier_stokes_exact.py
======================
基于种子项目 788_navier_stokes_3d_exact 的物理驱动模块。
提供三维不可压缩 Navier-Stokes 方程的 Ethier 精确解、Burgers 解与 Poiseuille 解，
并计算速度场与压力场的 PDE 残差，用于物理信息生成对抗网络（PI-GAN）的物理损失项。

核心公式：
  连续性方程：∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
  动量方程：∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
  其中 ν = μ/ρ 为运动粘性系数。

Ethier 精确解（C. Ross Ethier & David Steinman, 1994）：
  给定参数 a, d，速度分量与压力为：
    u = -a·(exp(a·x)·sin(a·y+d·z) + exp(a·z)·cos(a·x+d·y))·exp(-d²·t)
    v = -a·(exp(a·y)·sin(a·z+d·x) + exp(a·x)·cos(a·y+d·z))·exp(-d²·t)
    w = -a·(exp(a·z)·sin(a·x+d·y) + exp(a·y)·cos(a·z+d·x))·exp(-d²·t)
    p = 0.5·a²·exp(-2·d²·t)·(exp(2·a·x) + exp(2·a·y) + exp(2·a·z)
          + 2·sin(a·x+d·y)·cos(a·z+d·x)·exp(a·(y+z))
          + 2·sin(a·y+d·z)·cos(a·x+d·y)·exp(a·(z+x))
          + 2·sin(a·z+d·x)·cos(a·y+d·z)·exp(a·(x+y)))
"""

import numpy as np


def uvwp_ethier(a: float, d: float, x: np.ndarray, y: np.ndarray,
                z: np.ndarray, t: np.ndarray) -> tuple:
    """
    计算 Ethier 精确解在任意时空点处的速度分量 (u, v, w) 与压力 p。

    Parameters
    ----------
    a, d : float
        问题参数，典型取值为 a = π/4, d = π/2。
    x, y, z, t : np.ndarray
        空间与时间坐标，形状一致。

    Returns
    -------
    u, v, w, p : np.ndarray
        速度分量与压力场，形状与输入一致。
    """
    # 边界与数值鲁棒性处理
    a = float(a)
    d = float(d)
    if a == 0.0 and d == 0.0:
        raise ValueError("参数 a 与 d 不能同时为零，否则退化为 trivial 解。")

    # 指数项，限制幅值防止上溢
    ax = a * x
    ay = a * y
    az = a * z
    ex = np.exp(np.clip(ax, -700.0, 700.0))
    ey = np.exp(np.clip(ay, -700.0, 700.0))
    ez = np.exp(np.clip(az, -700.0, 700.0))

    e2t = np.exp(np.clip(-d * d * t, -700.0, 700.0))
    exy = np.exp(np.clip(a * (x + y), -700.0, 700.0))
    eyz = np.exp(np.clip(a * (y + z), -700.0, 700.0))
    ezx = np.exp(np.clip(a * (z + x), -700.0, 700.0))

    # 三角函数项
    sxy = np.sin(ax + d * y)
    syz = np.sin(ay + d * z)
    szx = np.sin(az + d * x)
    cxy = np.cos(ax + d * y)
    cyz = np.cos(ay + d * z)
    czx = np.cos(az + d * x)

    u = -a * (ex * syz + ez * cxy) * e2t
    v = -a * (ey * szx + ex * cyz) * e2t
    w = -a * (ez * sxy + ey * czx) * e2t
    p = 0.5 * a * a * e2t * e2t * (
        ex * ex
        + 2.0 * sxy * czx * eyz
        + ey * ey
        + 2.0 * syz * cxy * ezx
        + ez * ez
        + 2.0 * szx * cyz * exy
    )
    return u, v, w, p


def ns_residual(u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray,
                x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray,
                nu: float = 1.0, rho: float = 1.0) -> dict:
    """
    使用中心差分计算 Navier-Stokes 方程的连续性残差与动量残差。
    适用于结构化网格上的离散场数据。

    Parameters
    ----------
    u, v, w, p : np.ndarray, shape (nx, ny, nz) 或展平向量
        速度分量与压力场。
    x, y, z, t : np.ndarray
        对应坐标。若 x, y, z 为一维单调网格，则自动计算 dx, dy, dz。
    nu : float
        运动粘性系数 ν。
    rho : float
        密度 ρ。

    Returns
    -------
    dict
        包含 'continuity', 'momentum_x', 'momentum_y', 'momentum_z' 的残差范数字典。
    """
    # 若输入为一维展平向量，尝试重构为规则网格（假设为立方体根）
    if u.ndim == 1:
        n = int(round(u.size ** (1.0 / 3.0)))
        if n * n * n != u.size:
            # 非立方体网格：退化为简单前向/后向差分（间距取平均）
            return _ns_residual_flat(u, v, w, p, x, y, z, t, nu, rho)
        u = u.reshape((n, n, n))
        v = v.reshape((n, n, n))
        w = w.reshape((n, n, n))
        p = p.reshape((n, n, n))
        # 假设 x, y, z 为一维坐标
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        z = np.asarray(z).ravel()

    nx, ny, nz = u.shape
    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError("网格维度至少为 3 才能使用中心差分。")

    dx = float(x[1] - x[0]) if hasattr(x, '__len__') and len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if hasattr(y, '__len__') and len(y) > 1 else 1.0
    dz = float(z[1] - z[0]) if hasattr(z, '__len__') and len(z) > 1 else 1.0

    if dx == 0.0 or dy == 0.0 or dz == 0.0:
        raise ValueError("网格间距不能为零。")

    # 中心差分算子
    def dudx(f):
        return (f[2:, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (2.0 * dx)

    def dudy(f):
        return (f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1]) / (2.0 * dy)

    def dudz(f):
        return (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2]) / (2.0 * dz)

    def d2udx2(f):
        return (f[2:, 1:-1, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / (dx * dx)

    def d2udy2(f):
        return (f[1:-1, 2:, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / (dy * dy)

    def d2udz2(f):
        return (f[1:-1, 1:-1, 2:] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / (dz * dz)

    uc = u[1:-1, 1:-1, 1:-1]
    vc = v[1:-1, 1:-1, 1:-1]
    wc = w[1:-1, 1:-1, 1:-1]
    pc = p[1:-1, 1:-1, 1:-1]

    # 连续性残差：∂u/∂x + ∂v/∂y + ∂w/∂z
    div_u = dudx(u) + dudy(v) + dudz(w)
    res_continuity = float(np.mean(div_u ** 2))

    # 动量残差（忽略时间项，假设稳态或 t 已冻结）
    # R_x = u·∂u/∂x + v·∂u/∂y + w·∂u/∂z + (1/ρ)·∂p/∂x - ν·∇²u
    # TODO_HOLE_1_START: 实现 Navier-Stokes 动量方程残差计算
    # 提示：使用中心差分算子 dudx, dudy, dudz, d2udx2, d2udy2, d2udz2
    # 分别计算 x, y, z 三个方向的动量残差：
    #   conv = uc * dudx(u) + vc * dudy(u) + wc * dudz(u)   (对流项)
    #   press = (1.0 / rho) * dudx(p)                        (压力梯度项)
    #   visc = nu * (d2udx2(u) + d2udy2(u) + d2udz2(u))      (粘性扩散项)
    #   res = conv + press - visc                            (动量残差)
    # 最终返回字典需包含 continuity, momentum_x, momentum_y, momentum_z, total
    # TODO_HOLE_1_END

    return {
        "continuity": res_continuity,
        "momentum_x": 0.0,
        "momentum_y": 0.0,
        "momentum_z": 0.0,
        "total": res_continuity,
    }


def _ns_residual_flat(u, v, w, p, x, y, z, t, nu, rho):
    """非规则网格退化为简单有限差分。"""
    n = u.size
    dx = float(np.mean(np.diff(np.sort(np.unique(x))))) if np.unique(x).size > 1 else 1.0
    dy = float(np.mean(np.diff(np.sort(np.unique(y))))) if np.unique(y).size > 1 else 1.0
    dz = float(np.mean(np.diff(np.sort(np.unique(z))))) if np.unique(z).size > 1 else 1.0
    # 前向差分近似
    dudx = np.empty_like(u)
    dudx[:-1] = (u[1:] - u[:-1]) / dx
    dudx[-1] = dudx[-2]
    dvdy = np.empty_like(v)
    dvdy[:-1] = (v[1:] - v[:-1]) / dy
    dvdy[-1] = dvdy[-2]
    dwdz = np.empty_like(w)
    dwdz[:-1] = (w[1:] - w[:-1]) / dz
    dwdz[-1] = dwdz[-2]
    div_u = dudx + dvdy + dwdz
    # 动量残差仅做一阶估计
    res = np.abs(div_u)
    return {
        "continuity": float(np.mean(res ** 2)),
        "momentum_x": 0.0,
        "momentum_y": 0.0,
        "momentum_z": 0.0,
        "total": float(np.mean(res ** 2)),
    }


def generate_training_data(nx: int = 8, ny: int = 8, nz: int = 8,
                           a: float = np.pi / 4.0, d: float = np.pi / 2.0,
                           t_val: float = 0.05) -> tuple:
    """
    基于 Ethier 精确解生成结构化训练样本 (x, y, z, t, u, v, w, p)。

    Returns
    -------
    X : np.ndarray, shape (N, 4)
        时空坐标 [x, y, z, t]。
    Y : np.ndarray, shape (N, 4)
        物理量 [u, v, w, p]。
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing='ij')
    Tg = np.full_like(Xg, t_val)
    u, v, w, p = uvwp_ethier(a, d, Xg, Yg, Zg, Tg)
    N = nx * ny * nz
    X = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel(), Tg.ravel()])
    Y = np.column_stack([u.ravel(), v.ravel(), w.ravel(), p.ravel()])
    return X, Y
