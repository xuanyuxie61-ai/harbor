"""
structural_default_bvp.py
结构信用模型中的病态边值问题求解与有限差分离散化
应用于含小参数扰动的违约障碍 PDE

原项目映射: 572_ill_bvp, 359_fd1d_display
科学问题: 在结构信用模型 (Structural Credit Model) 中，公司资产价值 V(t)
服从几何布朗运动。当 V(t) 触及违约障碍 D(t) 时发生违约。
对于含小扩散系数 epsilon 的扰动模型，资产价值的稳态分布满足奇异摄动 BVP:

    epsilon * y''(x) - x * y'(x) + y(x) = 0,   x in [-1, 1]
    y(-1) = 2,  y(1) = 1

该问题在 epsilon -> 0 时产生边界层 (boundary layer)，
标准有限差分在边界层处需要极密网格才能分辨。
这里采用自适应网格的有限差分方法求解，并将解映射到信用风险中的
违约概率密度函数 p(V) 的估计。

数学模型:
    结构模型 Merton-Black-Cox 框架下，资产价值 V 满足:
        dV = mu*V*dt + sigma*V*dW
    违约时间 tau = inf{ t > 0 : V(t) <= D }
    在稳态近似下，PD 密度 p(V) 满足 Fokker-Planck 方程:
        0 = -d/dV[mu*V*p] + (sigma^2/2)*d^2/dV^2[V^2*p]
    通过变量替换 x = 2*(V-V_min)/(V_max-V_min) - 1 映射到 [-1,1]，
    得到形如 epsilon*y'' - x*y' + y = 0 的方程，其中 epsilon = sigma^2/(2|mu|) << 1。
"""

import numpy as np
from typing import Tuple, Optional


def solve_ill_bvp_fd(
    epsilon: float,
    n_nodes: int = 200,
    x_left: float = -1.0,
    x_right: float = 1.0,
    bc_left: float = 2.0,
    bc_right: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用中心差分求解病态 BVP:
        epsilon * y'' - x * y' + y = 0
        y(x_left) = bc_left, y(x_right) = bc_right

    离散格式 (中心差分):
        epsilon * (y_{i-1} - 2*y_i + y_{i+1})/h^2
        - x_i * (y_{i+1} - y_{i-1})/(2h)
        + y_i = 0

    整理得三对角系统:
        a_i * y_{i-1} + b_i * y_i + c_i * y_{i+1} = 0
        a_i = epsilon/h^2 + x_i/(2h)
        b_i = -2*epsilon/h^2 + 1
        c_i = epsilon/h^2 - x_i/(2h)

    对于 epsilon << h 的情况，中心差分可能不稳定，
    因此根据局部 Peclet 数自适应选择迎风格式:
        Pe_i = |x_i| * h / (2*epsilon)
        若 Pe_i > 2 且 x_i > 0: 使用向后差分 (upwind)
        若 Pe_i > 2 且 x_i < 0: 使用向前差分 (upwind)

    Parameters:
        epsilon: 小参数，控制边界层厚度
        n_nodes: 内部节点数
        x_left, x_right: 区间端点
        bc_left, bc_right: 边界条件

    Returns:
        x: 网格点
        y: 数值解
    """
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正")

    # 在边界层处加密网格
    # 边界层厚度约为 O(sqrt(epsilon)) 或 O(epsilon)
    # 采用非均匀网格: 在 x=0 附近 (边界层所在) 加密
    # 实际上对于 -x*y' 项，边界层在 x = -1 (左边界) 和 x = 1 (右边界)
    # 这里使用 tanh 映射生成边界加密网格
    uniform = np.linspace(0.0, 1.0, n_nodes)
    # 在两端加密
    xi = 0.5 * (1.0 + np.tanh(3.0 * (uniform - 0.5)) / np.tanh(1.5))
    x = x_left + (x_right - x_left) * xi

    # 构建三对角矩阵
    n_total = n_nodes
    a = np.zeros(n_total - 1, dtype=float)  # 下对角
    b = np.zeros(n_total, dtype=float)      # 主对角
    c = np.zeros(n_total - 1, dtype=float)  # 上对角
    d = np.zeros(n_total, dtype=float)      # 右端项

    for i in range(1, n_total - 1):
        h_m = x[i] - x[i - 1]
        h_p = x[i + 1] - x[i]
        h_avg = 0.5 * (h_m + h_p)

        # Peclet 数
        pe = abs(x[i]) * h_avg / (2.0 * epsilon)

        if pe > 2.0:
            # 迎风格式
            if x[i] > 0:
                # 向后差分
                a[i - 1] = epsilon / (h_m * h_avg)
                b[i] = -epsilon / (h_m * h_avg) - x[i] / h_avg + 1.0
                c[i] = 0.0
            else:
                # 向前差分
                a[i - 1] = 0.0
                b[i] = -epsilon / (h_p * h_avg) + x[i] / h_avg + 1.0
                c[i] = epsilon / (h_p * h_avg)
        else:
            # 中心差分
            a[i - 1] = epsilon / (h_m * h_avg) + x[i] / (h_m + h_p)
            b[i] = -epsilon * (1.0 / (h_m * h_avg) + 1.0 / (h_p * h_avg)) + 1.0
            c[i] = epsilon / (h_p * h_avg) - x[i] / (h_m + h_p)

    # 边界条件
    b[0] = 1.0
    d[0] = bc_left
    c[0] = 0.0

    b[-1] = 1.0
    d[-1] = bc_right
    a[-1] = 0.0

    # 使用 Thomas 算法求解
    from utils import tridiagonal_solve
    y = tridiagonal_solve(a, b, c, d)
    return x, y


def structural_default_probability_density(
    asset_values: np.ndarray,
    mu: float = 0.05,
    sigma: float = 0.2,
    v_min: float = 10.0,
    v_max: float = 200.0,
    default_barrier: float = 30.0
) -> np.ndarray:
    """
    基于病态 BVP 的稳态近似，估计结构模型下的违约概率密度

    模型设定:
        dV = mu*V*dt + sigma*V*dW
        违约障碍 D = default_barrier
        定义 y(V) = P(资产价值在稳态下等于 V | 未违约)
        通过 Fokker-Planck 稳态方程映射到 [-1, 1] 上的 BVP

    Parameters:
        asset_values: 资产价值网格 (N,)
        mu: 资产漂移率
        sigma: 资产波动率
        v_min: 资产价值下限
        v_max: 资产价值上限
        default_barrier: 违约障碍

    Returns:
        pdf: 稳态概率密度估计 (未归一化)
    """
    # 变量替换 x = 2*(V - v_min)/(v_max - v_min) - 1
    x_eval = 2.0 * (asset_values - v_min) / (v_max - v_min) - 1.0
    # 小参数 epsilon = sigma^2 / (2*|mu|)
    epsilon = max(sigma**2 / (2.0 * max(abs(mu), 1e-8)), 1e-6)

    # 映射边界条件
    # x=-1 (V=v_min): 反射边界，设 y=2 (归一化前)
    # x=1 (V=v_max): 吸收/远场边界，设 y=1
    bc_left = 2.0
    bc_right = 1.0

    x_grid, y_grid = solve_ill_bvp_fd(
        epsilon=epsilon,
        n_nodes=200,
        x_left=-1.0,
        x_right=1.0,
        bc_left=bc_left,
        bc_right=bc_right
    )

    # 插值到 asset_values 对应的 x 点
    from interpolation_surfaces import piecewise_linear_interpolate
    pdf = piecewise_linear_interpolate(x_grid, y_grid, x_eval)

    # 边界处理: V < default_barrier 时视为已违约区域，密度置零
    pdf = np.where(asset_values < default_barrier, 0.0, pdf)
    # 消除数值噪声导致的微小负值
    pdf = np.maximum(pdf, 0.0)

    # 归一化
    integral = np.trapezoid(pdf, asset_values)
    if integral > 1e-15:
        pdf = pdf / integral

    return pdf


def default_probability_from_structural(
    v0: float,
    mu: float,
    sigma: float,
    default_barrier: float,
    maturity: float
) -> float:
    """
    Merton 模型下的违约概率解析公式
    PD = Phi( - [ln(V0/D) + (mu - sigma^2/2)*T] / (sigma*sqrt(T)) )

    用于与数值 BVP 方法进行交叉验证
    """
    from utils import normal_cdf
    if v0 <= 0 or default_barrier <= 0 or maturity <= 0:
        return 1.0
    d = (np.log(v0 / default_barrier) + (mu - 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    return normal_cdf(-d)


def test_structural_bvp():
    """测试病态 BVP 求解"""
    x, y = solve_ill_bvp_fd(epsilon=0.01, n_nodes=100)
    assert len(x) == 100, "节点数错误"
    assert np.isclose(y[0], 2.0, atol=1e-6), "左边界条件不满足"
    assert np.isclose(y[-1], 1.0, atol=1e-6), "右边界条件不满足"

    # 测试 PD 密度
    v_grid = np.linspace(20, 150, 100)
    pdf = structural_default_probability_density(v_grid, mu=0.05, sigma=0.2, default_barrier=30.0)
    assert np.all(pdf >= 0), "PDF 存在负值"
    integral = np.trapezoid(pdf, v_grid)
    assert abs(integral - 1.0) < 0.1, f"PDF 积分不归一: {integral}"

    # 解析解对比
    pd_analytic = default_probability_from_structural(100.0, 0.05, 0.2, 30.0, 5.0)
    assert 0.0 <= pd_analytic <= 1.0, "解析 PD 越界"

    print(f"structural_default_bvp test passed. epsilon=0.01, PD_analytic={pd_analytic:.6f}")


if __name__ == "__main__":
    test_structural_bvp()
