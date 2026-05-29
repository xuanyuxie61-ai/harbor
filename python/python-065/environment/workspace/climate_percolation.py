"""
climate_percolation.py

基于 865_percolation_simulation 核心算法的极端天气渗流分析模块。

原项目使用二维渗流模拟来研究格点系统的连通性。
在本气候归因框架中，我们将极端降水/温度异常事件映射为格点上的
"被占据"状态，利用渗流理论识别跨越性极端事件簇（spanning clusters），
这些簇对应现实中具有大尺度空间连续性的极端天气系统。

核心科学公式：
- 占据概率：p = N_occupied / N_total
- 渗流阈值：在二维方格上，p_c ≈ 0.592746（ site percolation ）
- 关联长度：ξ(p) ~ |p - p_c|^{-ν}，其中 ν = 4/3（二维临界指数）
- 跨越概率：Π(p, L) 随系统尺寸 L 的变化满足有限尺寸标度：
    Π(p, L) = Φ( (p - p_c) * L^{1/ν} )
"""

import numpy as np


def detect_extreme_grid(anomaly_field, threshold):
    """
    将气候异常场二值化为极端事件格点。

    Parameters
    ----------
    anomaly_field : ndarray, shape (m, n)
        标准化后的气候异常场（如标准化降水指数 SPI）。
    threshold : float
        极端事件阈值（如 2.0 表示超过 2 个标准差）。

    Returns
    -------
    occupied : ndarray, shape (m, n)
        二值场，1 表示极端事件，0 表示非极端。
    """
    if anomaly_field.ndim != 2:
        raise ValueError("异常场必须是二维数组")
    occupied = np.zeros_like(anomaly_field, dtype=np.int64)
    occupied[anomaly_field > threshold] = 1
    return occupied


def components_2d(a):
    """
    二维连通分量标记（基于 865 的 components_2d 算法）。

    采用 4-邻域连通性（上下左右），标记所有连通的极端事件簇。
    使用栈实现深度优先搜索。

    Parameters
    ----------
    a : ndarray, shape (m, n)
        二值数组（0/1）。

    Returns
    -------
    c : ndarray, shape (m, n)
        连通分量标记数组，0 表示背景，1,2,... 表示不同簇。
    """
    m, n = a.shape
    c = np.zeros((m, n), dtype=np.int64)
    component_index = 0

    for ii in range(m):
        for jj in range(n):
            if a[ii, jj] != 0 and c[ii, jj] == 0:
                # 新连通分量
                component_index += 1
                stack = [(ii, jj)]
                while stack:
                    i, j = stack.pop()
                    if c[i, j] != 0:
                        continue
                    c[i, j] = component_index
                    # 4-邻域
                    if i - 1 >= 0 and a[i - 1, j] != 0 and c[i - 1, j] == 0:
                        stack.append((i - 1, j))
                    if i + 1 < m and a[i + 1, j] != 0 and c[i + 1, j] == 0:
                        stack.append((i + 1, j))
                    if j - 1 >= 0 and a[i, j - 1] != 0 and c[i, j - 1] == 0:
                        stack.append((i, j - 1))
                    if j + 1 < n and a[i, j + 1] != 0 and c[i, j + 1] == 0:
                        stack.append((i, j + 1))
    return c


def spanning_analysis(cls, m, n):
    """
    分析跨越性连通分量。

    在气候归因中，跨越性簇（从左到右或从上到下贯通整个区域）
    代表了具有行星尺度影响的大规模极端事件。

    Parameters
    ----------
    cls : ndarray
        连通分量标记数组。
    m, n : int
        网格尺寸。

    Returns
    -------
    spanx : int
        水平跨越分量数量。
    spany : int
        垂直跨越分量数量。
    component_sizes : list
        各分量大小列表。
    """
    component_num = int(cls.max())
    if component_num == 0:
        return 0, 0, []

    component_sizes = []
    isspanx = np.zeros(component_num + 1, dtype=np.int64)
    isspany = np.zeros(component_num + 1, dtype=np.int64)

    for comp in range(1, component_num + 1):
        size = int(np.sum(cls == comp))
        component_sizes.append(size)

        # 检查水平跨越
        left = np.any(cls[:, 0] == comp)
        right = np.any(cls[:, n - 1] == comp)
        if left and right:
            isspanx[comp] = 1

        # 检查垂直跨越
        top = np.any(cls[0, :] == comp)
        bottom = np.any(cls[m - 1, :] == comp)
        if top and bottom:
            isspany[comp] = 1

    spanx = int(np.sum(isspanx))
    spany = int(np.sum(isspany))
    return spanx, spany, component_sizes


def percolation_order_parameter(component_sizes, total_sites):
    """
    计算渗流序参量 P_∞ = S_max / N_total。

    在相变理论中，当 p > p_c 时 P_∞ > 0，表示出现了宏观连通分量。
    """
    if not component_sizes:
        return 0.0
    s_max = max(component_sizes)
    return s_max / total_sites


def correlation_length_estimate(component_sizes, threshold_bins=20):
    """
    从簇尺寸分布估计关联长度。

    关联长度 ξ 反映了极端事件的空间相关尺度，定义为：
        ξ^2 = (2 * sum_s s^2 n_s) / (sum_s s n_s)
    其中 n_s 是尺寸为 s 的簇的数量。
    """
    # [HOLE 1] 关联长度估计的科学公式实现被移除
    # 需要基于簇尺寸分布计算关联长度
    raise NotImplementedError("关联长度估计公式待实现")


def run_percolation_attribution(anomaly_field, threshold=2.0):
    """
    执行完整的渗流归因分析。

    Returns
    -------
    dict
        包含占据概率、跨越分量数、序参量、关联长度等指标。
    """
    m, n = anomaly_field.shape
    occupied = detect_extreme_grid(anomaly_field, threshold)
    nosites = int(np.sum(occupied))
    posites = nosites / (m * n)

    cls = components_2d(occupied)
    spanx, spany, component_sizes = spanning_analysis(cls, m, n)
    # [HOLE 1] 序参量和关联长度计算被移除
    p_inf = 0.0
    xi = 0.0

    return {
        "occupied_grid": occupied,
        "components": cls,
        "nosites": nosites,
        "posites": posites,
        "spanx": spanx,
        "spany": spany,
        "component_sizes": component_sizes,
        "p_infinity": p_inf,
        "correlation_length": xi,
    }


def test_percolation():
    np.random.seed(42)
    field = np.random.randn(20, 20)
    result = run_percolation_attribution(field, threshold=1.5)
    assert result["posites"] >= 0.0
    print("climate_percolation 自测试通过")


if __name__ == "__main__":
    test_percolation()
