r"""
causal_ode_dynamics.py
================================================================================
基于 Runge-Kutta 方法的因果效应动态演化与干预扩散模拟

原项目映射:
- 1036_rk4 — 四阶 Runge-Kutta 常微分方程数值积分
- 066_ball_distance — 单位球内随机点距离统计与蒙特卡洛采样

科学背景
--------
在动态因果推断（Dynamic Causal Inference）中，干预的效应不是瞬时的，
而是通过系统内部的耦合动力学随时间演化。我们将结构方程模型扩展为
动态因果模型（Dynamic Causal Model, DCM）：

$$ \frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, \mathbf{u}, \theta) $$

其中 $\mathbf{y}\in\mathbb{R}^p$ 为内生变量状态向量，$\mathbf{u}$ 为外生干预输入，
$\theta$ 为因果耦合参数。

对于线性近似，动态 SCM 可写为：
$$ \frac{d\mathbf{y}}{dt} = A \mathbf{y} + B \mathbf{u} + C \mathbf{y}\odot\mathbf{u} $$
其中 $A$ 为内生耦合矩阵，$B$ 为外生驱动矩阵，$C$ 为调制矩阵（ bilinear 项）。

核心公式
--------
1. 因果扩散方程（高维推广）：
   $$ \frac{\partial u}{\partial t} = D\nabla^2 u + \lambda u(1-u) + \sum_j \beta_j \delta(\mathbf{x}-\mathbf{x}_j) y_j $$
   其中 $D$ 为扩散系数，$\lambda$ 为 logistic 增长参数。

2. 四阶 Runge-Kutta 时间推进：
   $$ k_1 = \Delta t\, f(t_n, y_n) $$
   $$ k_2 = \Delta t\, f\left(t_n+\frac{\Delta t}{2}, y_n+\frac{k_1}{2}\right) $$
   $$ k_3 = \Delta t\, f\left(t_n+\frac{\Delta t}{2}, y_n+\frac{k_2}{2}\right) $$
   $$ k_4 = \Delta t\, f(t_n+\Delta t, y_n+k_3) $$
   $$ y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4) $$

3. 高维因果空间距离（Fisher 信息度量）：
   $$ d_{\text{causal}}^2(i,j) = (\theta_i - \theta_j)^T G(\theta) (\theta_i - \theta_j) $$
   其中 $G(\theta)$ 为 Fisher 信息矩阵。

4. 单位球内随机采样（蒙特卡洛估计因果空间体积）：
   对于 $p$ 维因果参数空间，在单位球 $B_p(1)$ 内均匀采样，
   估计因果效应的期望扩散范围：
   $$ \mathbb{E}[\|\Delta\mathbf{y}\|] \approx \frac{1}{N}\sum_{k=1}^{N}\|\mathbf{y}(t;\mathbf{\xi}_k) - \mathbf{y}_0\| $$
r"""

import numpy as np
from typing import Callable, Tuple, Optional


def rk4_integrate(f: Callable, t_span: Tuple[float, float],
                  y0: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    经典四阶 Runge-Kutta 方法求解 ODE 初值问题。

    Parameters
    ----------
    f : callable
        右端项函数 $f(t, y)$，返回 ndarray。
    t_span : (t0, tf)
        时间区间。
    y0 : ndarray
        初始条件。
    n_steps : int
        时间步数，必须 >= 1。

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
    y : ndarray, shape (n_steps+1, len(y0))
    r"""
    if n_steps < 1:
        raise ValueError("n_steps 必须至少为 1。")
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    t[0] = t0
    y[0, :] = y0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        k1 = dt * np.array(f(ti, yi))
        k2 = dt * np.array(f(ti + 0.5 * dt, yi + 0.5 * k1))
        k3 = dt * np.array(f(ti + 0.5 * dt, yi + 0.5 * k2))
        k4 = dt * np.array(f(ti + dt, yi + k3))
        y[i + 1, :] = yi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t[i + 1] = ti + dt
    return t, y


def ball_unit_sample_nd(dim: int) -> np.ndarray:
    r"""
    在单位球 $B_{\text{dim}}(1)$ 内均匀随机采样一点。

    方法：先生成标准正态向量，再归一化并乘以 $r^{1/\text{dim}}$，
    其中 $r\sim U(0,1)$。
    r"""
    if dim < 1:
        raise ValueError("维度必须至少为 1。")
    z = np.random.randn(dim)
    z = z / np.linalg.norm(z)
    u = np.random.rand()
    r = u ** (1.0 / dim)
    return r * z


def causal_ode_system(t: float, y: np.ndarray,
                       A: np.ndarray, B: np.ndarray,
                       u_func: Callable, bilinear: bool = False,
                       C: Optional[np.ndarray] = None) -> np.ndarray:
    r"""
    动态因果模型的 ODE 右端项：

    $$ \dot{y} = A y + B u(t) + C (y \odot u(t)) \quad \text{(若 bilinear=True)} $$

    Parameters
    ----------
    t : float
        当前时间。
    y : ndarray
        状态向量。
    A : ndarray
        内生耦合矩阵。
    B : ndarray
        外生驱动矩阵。
    u_func : callable
        外生输入函数 $u(t)$。
    bilinear : bool
        是否包含 bilinear 调制项。
    C : ndarray, optional
        调制矩阵。

    Returns
    -------
    dydt : ndarray
        时间导数。
    r"""
    u = np.array(u_func(t))
    dydt = A @ y + B @ u
    if bilinear and C is not None:
        dydt = dydt + C @ (y * u)
    return dydt


def simulate_intervention_diffusion(A: np.ndarray,
                                     B: np.ndarray,
                                     y0: np.ndarray,
                                     t_span: Tuple[float, float],
                                     n_steps: int = 200,
                                     intervention_time: float = 0.5,
                                     intervention_idx: Optional[int] = None,
                                     intervention_magnitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    模拟脉冲干预下因果效应的动态扩散过程。

    干预形式：在 intervention_time 时刻对第 intervention_idx 个变量
    施加脉冲输入 $u(t)=\delta(t-t_{\text{int}}) \cdot m$。
    数值上用一个短时间高斯脉冲近似狄拉克函数：
    $$ u(t) = m \cdot \exp\left(-\frac{(t-t_{\text{int}})^2}{2\sigma^2}\right) $$
    r"""
    p = A.shape[0]
    if intervention_idx is None:
        intervention_idx = 0
    if not (0 <= intervention_idx < p):
        raise ValueError("intervention_idx 超出范围。")

    sigma = (t_span[1] - t_span[0]) / n_steps * 3.0

    def u_func(t):
        u = np.zeros(p)
        u[intervention_idx] = intervention_magnitude * np.exp(-0.5 * ((t - intervention_time) / sigma) ** 2)
        return u

    def f(t, y):
        return causal_ode_system(t, y, A, B, u_func)

    t, y = rk4_integrate(f, t_span, y0, n_steps)
    return t, y


def monte_carlo_causal_distance(A: np.ndarray,
                                 B: np.ndarray,
                                 y0: np.ndarray,
                                 t_span: Tuple[float, float],
                                 n_steps: int = 100,
                                 n_samples: int = 200) -> Tuple[float, float]:
    r"""
    蒙特卡洛估计因果干预后系统状态的期望距离变化。

    对初始条件在单位球内随机扰动，运行 ODE，计算终态与无干预终态的距离。
    返回均值与方差。
    r"""
    p = len(y0)

    # 无干预基准
    def u_zero(t):
        return np.zeros(p)

    def f_base(t, y):
        return causal_ode_system(t, y, A, B, u_zero)

    _, y_base = rk4_integrate(f_base, t_span, y0, n_steps)
    y_final_base = y_base[-1, :]

    distances = np.zeros(n_samples)
    for k in range(n_samples):
        delta = ball_unit_sample_nd(p) * 0.1
        y0_pert = y0 + delta
        _, y_pert = rk4_integrate(f_base, t_span, y0_pert, n_steps)
        distances[k] = np.linalg.norm(y_pert[-1, :] - y_final_base)

    mu = float(np.mean(distances))
    var = float(np.var(distances, ddof=1)) if n_samples > 1 else 0.0
    return mu, var


def demo():
    r"""模块自测试。"""
    np.random.seed(13)
    p = 5
    # 构造因果耦合矩阵（下三角，符合 DAG 结构）
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            A[i, j] = 0.2 * np.random.randn()
    np.fill_diagonal(A, -0.5)  # 稳定化
    B = np.eye(p) * 0.8
    y0 = np.zeros(p)

    t, y = simulate_intervention_diffusion(A, B, y0, (0.0, 2.0), n_steps=200,
                                            intervention_time=0.5,
                                            intervention_idx=1,
                                            intervention_magnitude=2.0)
    print(f"[causal_ode_dynamics] 干预扩散模拟完成: t_end={t[-1]:.3f}, max_state={np.max(np.abs(y)):.4f}")

    mu, var = monte_carlo_causal_distance(A, B, y0, (0.0, 1.0), n_steps=100, n_samples=100)
    print(f"[causal_ode_dynamics] MC 因果距离: mean={mu:.4f}, var={var:.6f}")
    return t, y


if __name__ == "__main__":
    demo()
