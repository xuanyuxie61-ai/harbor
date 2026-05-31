"""
dynamics_integrator.py
======================
分子结构隐式松弛积分器

融合种子项目:
  - 063_backward_euler : 隐式后向 Euler ODE 求解器

科学背景:
  在预测分子性质前，通常需要将分子几何松弛至能量极小点（局部优化）。
  原子运动方程可写为阻尼 Langevin 方程的梯度流近似:
      dR/dt = -∇E(R)
  其中 E(R) 为分子势能。采用隐式后向 Euler:
      R_{n+1} = R_n + h * f(t_{n+1}, R_{n+1})
  即求解非线性方程:
      g(R_{n+1}) = R_{n+1} - R_n - h * (-∇E(R_{n+1})) = 0

  该方程通过 Newton-Raphson 迭代求解，保证了数值稳定性，
  即使采用较大步长 h 也不会失稳（A-稳定性）。
"""

import numpy as np
from typing import Callable, Tuple


def backward_euler_step(y0: np.ndarray, h: float,
                        f: Callable[[np.ndarray], np.ndarray],
                        df: Callable[[np.ndarray], np.ndarray],
                        max_iter: int = 20, tol: float = 1e-8) -> np.ndarray:
    """
    单步隐式后向 Euler:
        y1 = y0 + h * f(y1)
    用 Newton-Raphson 求解残差 g(y1) = y1 - y0 - h*f(y1) = 0。

    Newton 迭代:
        J_g = I - h * J_f(y1)
        y1^{k+1} = y1^{k} - J_g^{-1} g(y1^{k})

    Parameters
    ----------
    y0 : np.ndarray
        当前状态。
    h : float
        步长。
    f : callable
        右端项 f(y)。
    df : callable
        雅可比矩阵 J_f(y)（返回二维数组）。
    max_iter : int
        Newton 最大迭代次数。
    tol : float
        残差范数收敛阈值。

    Returns
    -------
    y1 : np.ndarray
        下一步状态。
    """
    y1 = y0.copy()
    dim = y0.size
    I = np.eye(dim, dtype=np.float64)

    for _ in range(max_iter):
        fy = f(y1)
        g = y1 - y0 - h * fy
        norm_g = np.linalg.norm(g)
        if norm_g < tol:
            break
        Jf = df(y1)
        Jg = I - h * Jf
        # 解线性系统
        try:
            dy = np.linalg.solve(Jg, -g)
        except np.linalg.LinAlgError:
            # 若奇异，采用伪逆
            dy = -np.linalg.lstsq(Jg, g, rcond=None)[0]
        y1 = y1 + dy
        # 阻尼：若步长过大则折半
        if np.linalg.norm(dy) > 10.0:
            y1 = y1 - 0.5 * dy
    return y1


def damped_gradient_flow(relax_coords: np.ndarray,
                        energy_func: Callable[[np.ndarray], float],
                        grad_func: Callable[[np.ndarray], np.ndarray],
                        hess_func: Callable[[np.ndarray], np.ndarray],
                        n_steps: int = 50,
                        h: float = 0.01,
                        tol: float = 1e-5) -> Tuple[np.ndarray, float]:
    """
    对分子坐标进行梯度流松弛至能量极小点。

    将坐标展平为向量 R，解:
        dR/dt = -∇E(R)
    每步用 backward_euler_step 隐式积分。

    Parameters
    ----------
    relax_coords : np.ndarray, shape (n_atoms, 3)
        初始原子坐标。
    energy_func : callable
        势能函数 E(R)。
    grad_func : callable
        梯度 ∇E(R)，返回 shape (n_atoms, 3)。
    hess_func : callable
        Hessian H(R)，返回 shape (3*n_atoms, 3*n_atoms)。
    n_steps : int
        积分步数。
    h : float
        伪时间步长。
    tol : float
        梯度范数收敛阈值。

    Returns
    -------
    coords_opt : np.ndarray
        优化后坐标。
    energy_opt : float
        优化后能量。
    """
    y = relax_coords.flatten().astype(np.float64)
    dim = y.size

    def f(y_vec):
        g = grad_func(y_vec.reshape(-1, 3))
        return -g.flatten()

    def df(y_vec):
        H = hess_func(y_vec.reshape(-1, 3))
        return -H

    for step in range(n_steps):
        y = backward_euler_step(y, h, f, df, max_iter=15, tol=1e-7)
        g_norm = np.linalg.norm(grad_func(y.reshape(-1, 3)))
        if g_norm < tol:
            break

    coords_opt = y.reshape(-1, 3)
    energy_opt = energy_func(coords_opt)
    return coords_opt, energy_opt


def lennard_jones_potential(coords: np.ndarray,
                            epsilon: float = 1.0,
                            sigma: float = 1.0) -> float:
    """
    Lennard-Jones 势能（用于演示结构松弛）:
        V_LJ(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]
    总能量为所有原子对之和。
    """
    n = coords.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(coords[i] - coords[j])
            r = max(r, 0.8 * sigma)
            sr6 = (sigma / r) ** 6
            sr12 = sr6 ** 2
            energy += 4.0 * epsilon * (sr12 - sr6)
    return energy


def lennard_jones_gradient(coords: np.ndarray,
                           epsilon: float = 1.0,
                           sigma: float = 1.0) -> np.ndarray:
    """Lennard-Jones 梯度。"""
    n = coords.shape[0]
    grad = np.zeros_like(coords)
    for i in range(n):
        for j in range(i + 1, n):
            dr = coords[i] - coords[j]
            r = np.linalg.norm(dr)
            r = max(r, 0.8 * sigma)
            sr6 = (sigma / r) ** 6
            sr12 = sr6 ** 2
            dVdr = 4.0 * epsilon * (-12.0 * sr12 / r + 6.0 * sr6 / r)
            g_vec = (dVdr / r) * dr
            grad[i] += g_vec
            grad[j] -= g_vec
    return grad


def lennard_jones_hessian(coords: np.ndarray,
                          epsilon: float = 1.0,
                          sigma: float = 1.0) -> np.ndarray:
    """
    Lennard-Jones Hessian 的有限差分近似。
    """
    n = coords.shape[0]
    dim = 3 * n
    H = np.zeros((dim, dim), dtype=np.float64)
    delta = 1e-5
    for idx in range(dim):
        coords_plus = coords.copy().flatten()
        coords_minus = coords.copy().flatten()
        coords_plus[idx] += delta
        coords_minus[idx] -= delta
        g_plus = lennard_jones_gradient(coords_plus.reshape(n, 3), epsilon, sigma).flatten()
        g_minus = lennard_jones_gradient(coords_minus.reshape(n, 3), epsilon, sigma).flatten()
        H[:, idx] = (g_plus - g_minus) / (2.0 * delta)
    # 对称化
    H = 0.5 * (H + H.T)
    return H
