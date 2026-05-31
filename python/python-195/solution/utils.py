"""
utils.py
通用工具函数与数值稳定性保障模块

包含：
- 数值稳定性边界检查
- 矩阵/向量辅助运算
- 网格数据处理（索引转换、面积计算等）
- 数学常数与物理参数
"""

import numpy as np
from typing import Tuple, Optional

# =============================================================================
# 物理常数与全局参数
# =============================================================================
EPSILON_MACHINE = np.finfo(float).eps
MAX_ITERATIONS = 10000
TOLERANCE_DEFAULT = 1e-10

# 等离子体物理相关参数（CGS单位制）
ELEMENTARY_CHARGE = 4.80320427e-10  # statcoulomb
ELECTRON_MASS = 9.10938356e-28      # g
BOLTZMANN_CONSTANT = 1.380649e-16   # erg/K


def safe_divide(a: np.ndarray, b: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """
    安全除法，避免除以零。
    
    数学表达:
        c_i = a_i / b_i   若 |b_i| > eps
        c_i = fallback    否则
    
    Parameters
    ----------
    a, b : np.ndarray
        被除数与除数数组
    fallback : float
        除数过小时的回退值
    
    Returns
    -------
    np.ndarray
        安全除法结果
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > EPSILON_MACHINE * 100.0
    result[mask] = a[mask] / b[mask]
    result[~mask] = fallback
    return result


def check_bounds(x: np.ndarray, x_min: float, x_max: float,
                 name: str = "variable") -> np.ndarray:
    """
    检查数组是否严格在边界内，对越界值进行截断并警告。
    
    Parameters
    ----------
    x : np.ndarray
        输入数组
    x_min, x_max : float
        下界与上界
    name : str
        变量名（用于日志）
    
    Returns
    -------
    np.ndarray
        截断后的数组
    """
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, x_min, x_max)
    violations = np.sum((x < x_min) | (x > x_max))
    if violations > 0:
        print(f"[WARNING] {name}: {violations} values out of bounds [{x_min}, {x_max}], clipped.")
    return x_clipped


def compute_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算二维三角形的有向面积（带符号）。
    
    数学公式（行列式形式）:
        A = 0.5 * | (x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1) |
    
    或等价地：
        A = 0.5 * | x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2) |
    
    Parameters
    ----------
    p1, p2, p3 : np.ndarray, shape (2,)
        三角形三个顶点坐标
    
    Returns
    -------
    float
        三角形有向面积。若面积过小（< 1e-14），返回 0.0 并警告。
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1])
                  - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    if abs(area) < 1e-14:
        print("[WARNING] Triangle area near zero; degenerate element detected.")
        return 0.0
    return area


def reference_to_physical_q4(q4: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """
    将参考四边形 [0,1]x[0,1] 上的点映射到物理四边形。
    
    基于双线性形函数（Q4单元）:
        N1(r,s) = (1-r)(1-s)
        N2(r,s) = r(1-s)
        N3(r,s) = rs
        N4(r,s) = (1-r)s
    
    物理坐标:
        x(r,s) = sum_{i=1}^{4} N_i(r,s) * X_i
        y(r,s) = sum_{i=1}^{4} N_i(r,s) * Y_i
    
    Parameters
    ----------
    q4 : np.ndarray, shape (4, 2)
        物理四边形四个顶点坐标（逆时针）
    rs : np.ndarray, shape (n, 2)
        参考坐标系中的点 (r, s) in [0,1]^2
    
    Returns
    -------
    np.ndarray, shape (n, 2)
        物理坐标
    """
    q4 = np.asarray(q4, dtype=float)
    rs = np.asarray(rs, dtype=float)
    n = rs.shape[0]
    r = rs[:, 0]
    s = rs[:, 1]

    # 边界检查
    r = check_bounds(r, 0.0, 1.0, "r")
    s = check_bounds(s, 0.0, 1.0, "s")

    psi = np.zeros((4, n))
    psi[0, :] = (1.0 - r) * (1.0 - s)
    psi[1, :] = r * (1.0 - s)
    psi[2, :] = r * s
    psi[3, :] = (1.0 - r) * s

    xy = q4.T @ psi  # shape (2, n)
    return xy.T


def mesh_base_one(element_node: np.ndarray, node_num: int) -> np.ndarray:
    """
    将网格元素节点索引转换为1-based（若原本为0-based）。
    
    边界检查:
        - 检测 node_min 和 node_max
        - 若 node_min == 0 且 node_max == node_num - 1，则执行 +1 转换
        - 若 node_min == 1 且 node_max == node_num，则保持不变
        - 否则抛出异常
    
    Parameters
    ----------
    element_node : np.ndarray, shape (element_order, element_num)
        元素节点连接矩阵
    node_num : int
        总节点数
    
    Returns
    -------
    np.ndarray
        1-based索引的元素节点矩阵
    """
    element_node = np.asarray(element_node, dtype=int)
    node_min = element_node.min()
    node_max = element_node.max()

    if node_min == 0 and node_max == node_num - 1:
        print("[INFO] Detected 0-based indexing; converting to 1-based.")
        return element_node + 1
    elif node_min == 1 and node_max == node_num:
        return element_node
    else:
        raise ValueError(
            f"Mesh indexing inconsistent: node_min={node_min}, node_max={node_max}, node_num={node_num}"
        )


def gauss_seidel_sweep(n: int, rhs: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    执行一次Gauss-Seidel迭代扫掠，并返回残差下降量。
    
    用于多重网格的光滑步骤。假设标准1D有限差分离散:
        A = tridiag(-1, 2, -1) / h^2
    
    迭代格式:
        x_i^{new} = (rhs_i + x_{i-1} + x_{i+1}) / 2
    
    Parameters
    ----------
    n : int
        未知数个数
    rhs : np.ndarray
        右端项（已缩放）
    x : np.ndarray
        当前解向量
    
    Returns
    -------
    x_new : np.ndarray
        更新后的解
    d : float
        解的变化量（无穷范数）
    """
    x = np.asarray(x, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float)
    if x.size < n or rhs.size < n:
        raise ValueError("Input vectors too short for given n.")

    x_new = x.copy()
    for i in range(1, n - 1):
        x_new[i] = 0.5 * (rhs[i] + x_new[i - 1] + x[i + 1])

    d = np.max(np.abs(x_new - x))
    return x_new, d


def restrict_coarse_to_fine(n_coarse: int, u_coarse: np.ndarray, n_fine: int,
                            u_fine: np.ndarray) -> np.ndarray:
    """
    将粗网格解延拓到细网格（线性插值）。
    
    延拓算子 I_{2h}^{h}:
        u_{2j}^{h}   = u_{j}^{2h}
        u_{2j+1}^{h} = 0.5 * (u_{j}^{2h} + u_{j+1}^{2h})
    
    Parameters
    ----------
    n_coarse : int
        粗网格节点数
    u_coarse : np.ndarray
        粗网格解
    n_fine : int
        细网格节点数（应为 2*(n_coarse-1)+1）
    u_fine : np.ndarray
        细网格解的初始/目标数组
    
    Returns
    -------
    np.ndarray
        延拓后的细网格解
    """
    u_coarse = np.asarray(u_coarse, dtype=float)
    u_fine = np.asarray(u_fine, dtype=float).copy()
    expected_fine = 2 * (n_coarse - 1) + 1
    if n_fine != expected_fine:
        raise ValueError(f"Fine grid size mismatch: expected {expected_fine}, got {n_fine}")
    if u_fine.size < n_fine or u_coarse.size < n_coarse:
        raise ValueError("Solution arrays too short.")

    for j in range(n_coarse - 1):
        u_fine[2 * j] = u_coarse[j]
        u_fine[2 * j + 1] = 0.5 * (u_coarse[j] + u_coarse[j + 1])
    u_fine[n_fine - 1] = u_coarse[n_coarse - 1]
    return u_fine


def restrict_fine_to_coarse(n_fine: int, u_fine: np.ndarray, rhs_fine: np.ndarray,
                            n_coarse: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将细网格残差限制到粗网格（完全加权限制）。
    
    限制算子 I_{h}^{2h}:
        r_{j}^{2h} = 0.25 * r_{2j-1}^{h} + 0.5 * r_{2j}^{h} + 0.25 * r_{2j+1}^{h}
    
    同时限制解的误差（即当前解本身用于粗网格修正）。
    
    Parameters
    ----------
    n_fine : int
        细网格节点数
    u_fine : np.ndarray
        细网格解
    rhs_fine : np.ndarray
        细网格右端项
    n_coarse : int
        粗网格节点数（应为 (n_fine-1)/2 + 1，要求 n_fine 为奇数）
    
    Returns
    -------
    u_coarse : np.ndarray
        粗网格解（初始猜测）
    rhs_coarse : np.ndarray
        粗网格右端项（限制后的残差）
    """
    u_fine = np.asarray(u_fine, dtype=float)
    rhs_fine = np.asarray(rhs_fine, dtype=float)
    expected_coarse = (n_fine - 1) // 2 + 1
    if n_coarse != expected_coarse:
        raise ValueError(f"Coarse grid size mismatch: expected {expected_coarse}, got {n_coarse}")
    if (n_fine - 1) % 2 != 0:
        raise ValueError("n_fine must be odd for standard restriction.")

    u_coarse = np.zeros(n_coarse)
    rhs_coarse = np.zeros(n_coarse)

    # 边界直接复制
    u_coarse[0] = u_fine[0]
    rhs_coarse[0] = rhs_fine[0]
    u_coarse[n_coarse - 1] = u_fine[n_fine - 1]
    rhs_coarse[n_coarse - 1] = rhs_fine[n_fine - 1]

    for j in range(1, n_coarse - 1):
        fine_idx = 2 * j
        rhs_coarse[j] = (
            0.25 * rhs_fine[fine_idx - 1]
            + 0.5 * rhs_fine[fine_idx]
            + 0.25 * rhs_fine[fine_idx + 1]
        )
        u_coarse[j] = u_fine[fine_idx]

    return u_coarse, rhs_coarse


def is_power_of_two(n: int) -> bool:
    """检查整数是否为2的幂次。"""
    return n > 0 and (n & (n - 1)) == 0
