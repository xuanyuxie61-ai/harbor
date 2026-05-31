"""
interpolation_surfaces.py
多维曲线插值与离散节点生成
应用于信用风险中的违约概率期限结构、信用利差曲面插值

原项目映射: 590_interp
科学问题: 在信用风险建模中，违约概率 (PD) 和信用利差 (CS) 通常是期限 T 的函数。
对于离散的市场观测点 {(T_i, PD_i)}，需要构造光滑的期限结构曲线 PD(T)。
此外，在多因子模型中，载荷矩阵 B 可能需要在离散参数点上插值。
本模块提供:
    - Lagrange 全局多项式插值
    - 分段线性插值
    - Clenshaw-Curtis / Fejér / Newton-Cotes 节点生成
    - 弧长参数化 (用于相关性曲面)
"""

import numpy as np
from typing import Tuple, Optional


def lagrange_basis(t_data: np.ndarray, j: int, t: float) -> float:
    """
    计算第 j 个 Lagrange 基函数在 t 处的值
    L_j(t) = prod_{m!=j} (t - t_m) / (t_j - t_m)
    """
    n = len(t_data)
    result = 1.0
    for m in range(n):
        if m != j:
            denom = t_data[j] - t_data[m]
            if abs(denom) < 1e-15:
                return 0.0
            result *= (t - t_data[m]) / denom
    return result


def lagrange_interpolate(t_data: np.ndarray, p_data: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """
    Lagrange 全局多项式插值
    p(t) = sum_j p_j * L_j(t)

    适用于信用利差期限结构的低阶光滑插值 (节点数 <= 10)。
    高阶时可能出现 Runge 现象。

    Parameters:
        t_data: 已知节点 (已排序)
        p_data: 节点值
        t_eval: 求值点

    Returns:
        插值结果
    """
    if len(t_data) != len(p_data):
        raise ValueError("节点与值维度不匹配")
    if len(t_data) < 2:
        raise ValueError("至少需要 2 个节点")

    result = np.zeros_like(t_eval, dtype=float)
    for j in range(len(t_data)):
        lj = np.array([lagrange_basis(t_data, j, t) for t in t_eval])
        result += p_data[j] * lj
    return result


def piecewise_linear_interpolate(t_data: np.ndarray, p_data: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """
    分段线性插值
    在区间 [t_i, t_{i+1}] 内使用凸组合:
        p(t) = p_i * (t_{i+1} - t)/(t_{i+1} - t_i) + p_{i+1} * (t - t_i)/(t_{i+1} - t_i)

    信用风险中的标准做法：对 PD 期限结构和回收率曲线使用分段线性插值
    以保证单调性和无振荡。
    """
    if len(t_data) != len(p_data):
        raise ValueError("节点与值维度不匹配")
    t_data = np.asarray(t_data)
    p_data = np.asarray(p_data)
    t_eval = np.asarray(t_eval)

    result = np.zeros_like(t_eval, dtype=float)
    for idx, t in enumerate(t_eval):
        if t <= t_data[0]:
            result[idx] = p_data[0]
        elif t >= t_data[-1]:
            result[idx] = p_data[-1]
        else:
            # 二分查找区间
            i = np.searchsorted(t_data, t) - 1
            i = max(0, min(i, len(t_data) - 2))
            dt = t_data[i + 1] - t_data[i]
            if abs(dt) < 1e-15:
                result[idx] = p_data[i]
            else:
                w = (t - t_data[i]) / dt
                result[idx] = p_data[i] * (1.0 - w) + p_data[i + 1] * w
    return result


def clenshaw_curtis_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    Clenshaw-Curtis 节点: x_i = cos((n-i)*pi/(n-1)), i=0,...,n-1
    在 [a, b] 上映射
    用于信用衍生品定价中的快速积分变换
    """
    if n < 2:
        return np.array([(a + b) / 2.0])
    i = np.arange(n)
    x = np.cos((n - 1 - i) * np.pi / (n - 1))
    # 映射到 [a, b]
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def fejer1_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    Fejér Type 1 节点: x_i = cos((2i+1)*pi/(2n))
    开区间 (-1, 1) 上的节点，不含端点
    """
    i = np.arange(n)
    x = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def fejer2_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    Fejér Type 2 节点: x_i = cos(i*pi/n)
    包含端点，用于闭区间积分
    """
    i = np.arange(n)
    x = np.cos(i * np.pi / (n - 1)) if n > 1 else np.array([0.0])
    if n == 1:
        return np.array([(a + b) / 2.0])
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def arc_length_parameterize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对多维点列进行弧长参数化
    t_k = sum_{i=1}^k ||p_i - p_{i-1}||
    返回参数 t 和累积距离 s

    在信用风险中用于对相关性矩阵流形上的离散样本进行均匀参数化
    """
    if points.ndim != 2:
        raise ValueError("points 必须为二维数组 (n_points x dim)")
    n = points.shape[0]
    s = np.zeros(n, dtype=float)
    for i in range(1, n):
        s[i] = s[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    # 归一化到 [0, 1]
    if s[-1] > 1e-15:
        t = s / s[-1]
    else:
        t = np.linspace(0.0, 1.0, n)
    return t, s


def interpolate_credit_curve(
    maturities: np.ndarray,
    values: np.ndarray,
    eval_maturities: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    信用曲线插值统一接口
    支持 "linear" 和 "lagrange"

    Parameters:
        maturities: 观测期限 (年)
        values: 观测值 (PD, spread, 或 recovery)
        eval_maturities: 求值期限
        method: 插值方法

    Returns:
        插值后的值
    """
    # 排序
    idx = np.argsort(maturities)
    maturities = maturities[idx]
    values = values[idx]

    # 边界处理：要求期限非负
    if np.any(maturities < 0):
        raise ValueError("期限不能为负")
    if np.any(eval_maturities < 0):
        raise ValueError("求值期限不能为负")

    if method.lower() == "linear":
        return piecewise_linear_interpolate(maturities, values, eval_maturities)
    elif method.lower() == "lagrange":
        # 限制节点数避免 Runge 现象
        if len(maturities) > 10:
            raise ValueError("Lagrange 插值节点数超过 10，建议使用分段线性")
        return lagrange_interpolate(maturities, values, eval_maturities)
    else:
        raise ValueError(f"不支持的插值方法: {method}")


def test_interpolation():
    """测试插值函数"""
    t_data = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    p_data = np.array([0.01, 0.015, 0.022, 0.028, 0.035])
    t_eval = np.array([0.5, 1.5, 4.0])

    p_lin = piecewise_linear_interpolate(t_data, p_data, t_eval)
    assert np.all(p_lin >= np.min(p_data)) and np.all(p_lin <= np.max(p_data)), "线性插值越界"

    p_lag = lagrange_interpolate(t_data, p_data, t_eval)
    # 在节点处应当精确恢复
    p_lag_nodes = lagrange_interpolate(t_data, p_data, t_data)
    assert np.allclose(p_lag_nodes, p_data), "Lagrange 插值节点恢复失败"

    # 测试弧长参数化
    pts = np.random.randn(10, 3)
    t_param, s = arc_length_parameterize(pts)
    assert len(t_param) == 10 and np.isclose(t_param[-1], 1.0), "弧长参数化错误"

    print("interpolation_surfaces test passed.")


if __name__ == "__main__":
    test_interpolation()
