"""
utils.py

通用工具与辅助算法模块

本模块融合以下种子项目的核心算法：
  - 1100_sort_rc: 外部可控排序（堆排序变体）

科学背景：
  在肿瘤微环境模拟中，经常需要对细胞群体按特定属性排序，
  例如按增殖速率、药物敏感性或空间坐标。
  sort_rc 实现了 Nijenhuis-Wilf 的外部排序接口，
  允许调用者自定义比较逻辑，适用于复杂对象的排序。

  此外提供数值鲁棒性工具：
    - safe_divide: 安全除法
    - validate_parameters: 参数边界检查
    - sigmoid: 标准 sigmoid 函数
"""

import numpy as np
from typing import Tuple, List


def sort_rc(n: int, indx: int, isgn: int,
            i_save: List[int], j_save: List[int],
            k_save: List[int], l_save: List[int],
            n_save: List[int]) -> Tuple[int, int, int]:
    """
    外部可控排序（Nijenhuis-Wilf 算法）。

    该函数不直接操作数据，而是通过 indx 信号控制排序流程：
      indx < 0: 请求比较元素 i 和 j，调用者需设置 isgn
      indx > 0: 请求交换元素 i 和 j
      indx = 0: 排序完成

    参数:
        n: 元素总数
        indx: 通信信号（输入上一次的输出值，首次调用为 0）
        isgn: 比较结果（i <= j 时 <=0，否则 >0）
        i_save, j_save, k_save, l_save, n_save: 持久状态列表（单元素）

    返回:
        indx_out, i, j
    """
    if n < 1:
        return 0, 0, 0

    if indx == 0:
        k_save[0] = n // 2
        l_save[0] = k_save[0]
        n_save[0] = n

    elif indx < 0:
        if indx == -2:
            if isgn < 0:
                i_save[0] += 1
            j_save[0] = l_save[0]
            l_save[0] = i_save[0]
            return -1, i_save[0], j_save[0]

        if isgn > 0:
            return 2, i_save[0], j_save[0]

        if k_save[0] <= 1:
            if n_save[0] == 1:
                i_save[0] = 0
                j_save[0] = 0
                return 0, 0, 0
            else:
                i_save[0] = n_save[0]
                n_save[0] -= 1
                j_save[0] = 1
                return 1, i_save[0], j_save[0]

        k_save[0] -= 1
        l_save[0] = k_save[0]

    elif indx == 1:
        l_save[0] = k_save[0]

    while True:
        i_save[0] = 2 * l_save[0]

        if i_save[0] == n_save[0]:
            j_save[0] = l_save[0]
            l_save[0] = i_save[0]
            return -1, i_save[0], j_save[0]
        elif i_save[0] <= n_save[0]:
            j_save[0] = i_save[0] + 1
            return -2, i_save[0], j_save[0]

        if k_save[0] <= 1:
            break

        k_save[0] -= 1
        l_save[0] = k_save[0]

    if n_save[0] == 1:
        i_save[0] = 0
        j_save[0] = 0
        return 0, 0, 0
    else:
        i_save[0] = n_save[0]
        n_save[0] -= 1
        j_save[0] = 1
        return 1, i_save[0], j_save[0]


def external_sort_array(arr: np.ndarray) -> np.ndarray:
    """
    使用 sort_rc 外部排序接口对数组进行排序。

    参数:
        arr: 输入数组

    返回:
        sorted_arr: 升序排列的数组
    """
    arr = arr.copy()
    n = arr.shape[0]
    if n <= 1:
        return arr

    i_save = [-1]
    j_save = [-1]
    k_save = [-1]
    l_save = [-1]
    n_save = [-1]

    indx = 0
    isgn = 0

    while True:
        indx, i, j = sort_rc(n, indx, isgn, i_save, j_save, k_save, l_save, n_save)
        if indx < 0:
            isgn = 1 if arr[i - 1] > arr[j - 1] else -1
        elif indx > 0:
            arr[i - 1], arr[j - 1] = arr[j - 1], arr[i - 1]
        else:
            break

    return arr


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    安全除法，防止除零。

    返回 a / max(|b|, eps) * sign(b)
    """
    b_safe = np.where(np.abs(b) < eps, np.sign(b) * eps if np.sign(b) != 0 else eps, b)
    return a / b_safe


def sigmoid(x: np.ndarray, steepness: float = 1.0, midpoint: float = 0.0) -> np.ndarray:
    """
    数值稳定的 sigmoid 函数。

    S(x) = 1 / (1 + exp(-steepness * (x - midpoint)))
    """
    z = steepness * (x - midpoint)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def validate_parameters(params: dict, bounds: dict) -> bool:
    """
    批量参数边界检查。

    参数:
        params: {name: value}
        bounds: {name: (low, high)}

    返回:
        True 如果所有参数在边界内，否则抛出 ValueError
    """
    for name, value in params.items():
        if name in bounds:
            low, high = bounds[name]
            if not (low <= value <= high):
                raise ValueError(
                    f"validate_parameters: 参数 {name}={value} 超出边界 [{low}, {high}]"
                )
    return True


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    计算 Gini 系数，衡量肿瘤内部资源分配不均程度。

    公式:
        G = (sum_i sum_j |x_i - x_j|) / (2 * n * sum_i x_i)

    参数:
        values: 非负数组

    返回:
        gini: [0, 1] 范围内的系数，0 为完全均匀，1 为完全不均
    """
    values = np.asarray(values, dtype=float)
    values = np.where(values < 0, 0.0, values)
    n = values.shape[0]
    if n == 0:
        return 0.0

    total = np.sum(values)
    if total < 1e-15:
        return 0.0

    # 排序后计算 Lorenz 曲线面积
    sorted_vals = np.sort(values)
    cumsum = np.cumsum(sorted_vals)
    B = np.sum(cumsum) / (n * total)
    return float(1.0 - 2.0 * B + 1.0 / n)


def morse_potential(r: np.ndarray, epsilon: float = 1.0,
                    r_eq: float = 1.0, alpha: float = 6.0) -> np.ndarray:
    """
    Morse 势函数，用于描述细胞间相互作用势能。

    V(r) = epsilon * [ (1 - exp(-alpha*(r - r_eq)))^2 - 1 ]

    参数:
        r: 距离数组
        epsilon: 势阱深度
        r_eq: 平衡距离
        alpha: 势阱宽度参数

    返回:
        V: 势能数组
    """
    r = np.asarray(r, dtype=float)
    dr = r - r_eq
    exp_term = np.exp(-alpha * dr)
    return epsilon * ((1.0 - exp_term) ** 2 - 1.0)
