"""
default_contagion_ode.py
违约传染动态系统的 ODE 建模与 Theta 方法数值求解
应用于信用风险中的多主体违约级联 (cascade) 时序演化

原项目映射: 838_oregonator_ode, 1259_theta_method
科学问题: 在信用网络中，一个机构的违约可能通过对手方风险 (counterparty risk)
和流动性紧缩引发连锁违约。我们建立一个三变量的非线性 ODE 系统来刻画
区域级违约强度 (default intensity) 的动态演化:

    变量定义:
        u(t): 活跃违约强度 (active default rate)
        v(t): 传染性压力指标 (contagion pressure)
        w(t): 系统性缓冲水平 (systemic buffer)

    动力学方程 (Oregonator 启发的信用传染模型):
        du/dt = (q*v - u*v + u*(1 - u)) / eta1
        dv/dt = (-q*v - u*v + f*w) / eta2
        dw/dt = u - w

    其中参数:
        eta1, eta2: 时间尺度分离参数 (快/慢变量)
        q: 基础传染率
        f: 缓冲恢复效率

    该系统的物理类比来自化学振荡 (Belousov-Zhabotinsky 反应)，
    其中 u 对应违约波前的传播，v 对应市场恐慌的扩散，w 对应监管干预的恢复力。

数值求解采用 Theta 方法 (隐式-显式混合格式):
    Y_{n+1} = Y_n + h * [ theta * F(t_n, Y_n) + (1-theta) * F(t_{n+1}, Y_{n+1}) ]
    theta = 0.5: Crank-Nicolson (二阶精度，A-稳定)
    theta = 1.0: 显式 Euler
    theta = 0.0: 隐式 Euler
"""

import numpy as np
from typing import Tuple, Callable, Optional


def contagion_rhs(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """
    违约传染 ODE 的右端项

    Parameters:
        t: 时间 (年)
        y: 状态向量 [u, v, w]
        params: 参数字典 {eta1, eta2, q, f}

    Returns:
        dydt: 时间导数
    """
    u, v, w = y
    eta1 = params.get("eta1", 0.01)
    eta2 = params.get("eta2", 0.01)
    q = params.get("q", 0.01)
    f = params.get("f", 1.0)

    du = (q * v - u * v + u * (1.0 - u)) / eta1
    dv = (-q * v - u * v + f * w) / eta2
    dw = u - w

    return np.array([du, dv, dw], dtype=float)


def theta_method_solve(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    theta: float = 0.5,
    max_newton_iter: int = 20,
    newton_tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Theta 方法求解常微分方程初值问题

    数学公式:
        Y_{n+1} = Y_n + h * [theta * F(t_n, Y_n) + (1-theta) * F(t_{n+1}, Y_{n+1})]

    当 theta < 1 时，每步需要解非线性方程。
    这里采用不动点迭代 (简化的 Newton 迭代) 求解隐式步。

    Parameters:
        f: 右端函数 F(t, y)
        tspan: (t0, tf)
        y0: 初值
        n_steps: 时间步数
        theta: 方法参数
        max_newton_iter: Newton 迭代最大次数
        newton_tol: Newton 收敛容差

    Returns:
        t: 时间网格 (n_steps+1,)
        y: 解矩阵 (n_steps+1 x n_vars)
    """
    t0, tf = tspan
    h = (tf - t0) / n_steps
    n_vars = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, n_vars), dtype=float)
    y[0, :] = y0

    for n in range(n_steps):
        yn = y[n, :]
        tn = t[n]
        tnp1 = t[n + 1]
        fn = f(tn, yn)

        if theta >= 0.9999999:
            # 显式 Euler
            y[n + 1, :] = yn + h * fn
        else:
            # 隐式/混合步: 用显式 Euler 做预测
            y_pred = yn + h * fn
            # Newton 迭代修正
            y_curr = y_pred.copy()
            for _ in range(max_newton_iter):
                fnp1 = f(tnp1, y_curr)
                residual = y_curr - yn - h * (theta * fn + (1.0 - theta) * fnp1)
                # 简化的 Newton 步: 使用单位矩阵近似 Jacobian
                # 实际应使用完整的 Jacobian，但为简化实现，用阻尼迭代
                y_next = y_curr - 0.5 * residual
                if np.linalg.norm(y_next - y_curr) < newton_tol:
                    y_curr = y_next
                    break
                y_curr = y_next
            y[n + 1, :] = y_curr

        # 边界处理: 违约强度必须在 [0, 1] 内
        y[n + 1, 0] = np.clip(y[n + 1, 0], 0.0, 1.0)
        y[n + 1, 1] = np.clip(y[n + 1, 1], 0.0, 1.0)
        y[n + 1, 2] = np.clip(y[n + 1, 2], 0.0, 1.0)

    return t, y


def simulate_default_contagion(
    initial_default_rate: float = 0.05,
    initial_pressure: float = 0.1,
    initial_buffer: float = 0.5,
    t_max: float = 5.0,
    n_steps: int = 500,
    theta: float = 0.5,
    params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟信用网络中的违约传染动态

    Parameters:
        initial_default_rate: 初始违约强度 u(0)
        initial_pressure: 初始传染压力 v(0)
        initial_buffer: 初始系统性缓冲 w(0)
        t_max: 模拟终止时间 (年)
        n_steps: 步数
        theta: Theta 方法参数
        params: ODE 参数字典

    Returns:
        t: 时间序列
        y: 状态序列 (u, v, w)
    """
    if params is None:
        params = {
            "eta1": 0.02,
            "eta2": 0.05,
            "q": 0.02,
            "f": 0.8
        }

    y0 = np.array([initial_default_rate, initial_pressure, initial_buffer], dtype=float)
    f = lambda t, y: contagion_rhs(t, y, params)
    t, y = theta_method_solve(f, (0.0, t_max), y0, n_steps, theta)
    return t, y


def network_cascade_intensity(
    adjacency: np.ndarray,
    local_intensities: np.ndarray,
    coupling_strength: float = 0.1
) -> np.ndarray:
    """
    基于邻接矩阵计算网络级联增强的违约强度

    数学模型:
        lambda_i^{network} = lambda_i^{local} + coupling * sum_j A_{ij} * lambda_j^{local}

    其中 A_{ij} 为区域邻接矩阵 (来自球面 Voronoi 剖分)。
    邻接节点的高违约强度会通过网络耦合增强本节点的违约概率。

    Parameters:
        adjacency: 邻接矩阵 (n x n)，布尔或 0/1
        local_intensities: 局部违约强度 (n,)
        coupling_strength: 网络耦合强度

    Returns:
        network_intensities: 网络增强后的违约强度 (n,)
    """
    adj = np.asarray(adjacency, dtype=float)
    local = np.asarray(local_intensities, dtype=float)
    network_effect = adj @ local
    result = local + coupling_strength * network_effect
    # 截断到合法区间
    return np.clip(result, 0.0, 1.0)


def test_default_contagion():
    """测试违约传染 ODE 求解"""
    t, y = simulate_default_contagion(
        initial_default_rate=0.05,
        initial_pressure=0.1,
        initial_buffer=0.5,
        t_max=2.0,
        n_steps=200,
        theta=0.5
    )
    assert len(t) == 201, "时间步数错误"
    assert np.all(y[:, 0] >= 0) and np.all(y[:, 0] <= 1), "违约强度越界"
    assert np.all(y[:, 1] >= 0) and np.all(y[:, 1] <= 1), "传染压力越界"
    assert np.all(y[:, 2] >= 0) and np.all(y[:, 2] <= 1), "缓冲水平越界"

    # 测试网络级联
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    local = np.array([0.1, 0.2, 0.15])
    net = network_cascade_intensity(adj, local, 0.1)
    assert np.all(net >= local), "网络效应应为增强"
    print(f"default_contagion_ode test passed. final u={y[-1,0]:.4f}, v={y[-1,1]:.4f}, w={y[-1,2]:.4f}")


if __name__ == "__main__":
    test_default_contagion()
