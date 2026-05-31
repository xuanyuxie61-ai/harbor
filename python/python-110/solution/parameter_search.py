"""
parameter_search.py - 量子点单光子源参数优化搜索模块

融合原项目 1057_satisfy_brute（穷举搜索）的核心思想，
对量子点-微腔系统的关键物理参数进行离散化网格搜索，
寻找使单光子品质（g^(2)(0) 最小、Purcell 增强最大）最优的参数组合。

搜索空间维度示例：
    - 量子点半径 R_dot
    - 腔衰减率 kappa
    - 耦合强度 g
    - 量子点失谐量 Delta = omega_dot - omega_c
"""

import numpy as np
from typing import Callable, Dict, List, Tuple
from utils import validate_array_1d


def int_to_binary_vector(i4: int, n_bits: int) -> np.ndarray:
    """
    将整数转换为 n_bits 维二进制向量（源自 i4_to_bvec）。
    
    用于编码参数组合为二进制串，便于穷举搜索。
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if i4 < 0:
        raise ValueError("i4 must be non-negative")
    bvec = np.zeros(n_bits, dtype=int)
    temp = i4
    for i in range(n_bits - 1, -1, -1):
        bvec[i] = temp % 2
        temp = temp // 2
    return bvec


def brute_force_optimize(
    parameter_ranges: Dict[str, Tuple[float, float, int]],
    objective: Callable[[Dict[str, float]], float],
    constraint: Callable[[Dict[str, float]], bool] = None,
) -> Dict[str, any]:
    """
    对多维参数空间进行穷举网格搜索（源自 satisfy_brute）。
    
    参数:
        parameter_ranges: 参数字典，每个值为 (min, max, n_grid_points)
        objective: 目标函数，输入参数字典，输出标量（越小越好）
        constraint: 可选约束函数，返回 True 表示参数可行
    
    返回:
        best_params: 最优参数组合
        best_value: 最优目标函数值
        all_results: 所有可行参数及对应目标值
    """
    param_names = list(parameter_ranges.keys())
    grids = []
    for name in param_names:
        pmin, pmax, n_pts = parameter_ranges[name]
        if n_pts < 2:
            raise ValueError(f"Parameter {name} needs at least 2 grid points")
        grid = np.linspace(pmin, pmax, n_pts)
        grids.append(grid)

    # 计算总组合数
    total_combinations = 1
    for g in grids:
        total_combinations *= g.size

    best_value = float('inf')
    best_params = None
    all_results = []

    # 递归生成网格点并评估
    def recursive_search(depth: int, current_params: Dict[str, float]):
        nonlocal best_value, best_params
        if depth == len(param_names):
            if constraint is not None and not constraint(current_params):
                return
            val = objective(current_params)
            all_results.append((current_params.copy(), val))
            if val < best_value:
                best_value = val
                best_params = current_params.copy()
            return
        name = param_names[depth]
        for val in grids[depth]:
            current_params[name] = float(val)
            recursive_search(depth + 1, current_params)

    recursive_search(0, {})
    return {
        "best_params": best_params,
        "best_value": best_value,
        "all_results": all_results,
        "total_evaluated": len(all_results),
    }


def single_photon_figure_of_merit(
    g2_0: float,
    purcell_factor: float,
    extraction_efficiency: float,
    dephasing_rate: float,
    target_dephasing: float = 1e9,
) -> float:
    """
    单光子源综合品质因数（Figure of Merit, FoM）：
    
        FoM = - log10( g^(2)(0) + epsilon ) + w_p * log10(F_p) + w_eta * eta_extraction
                - w_d * (dephasing / target_dephasing)^2
    
    值越大表示品质越好。在搜索中转化为最小化 -FoM。
    """
    eps = 1e-15
    w_p = 1.0
    w_eta = 2.0
    w_d = 1.0
    if g2_0 < 0 or purcell_factor < 0 or extraction_efficiency < 0 or dephasing_rate < 0:
        raise ValueError("All physical quantities must be non-negative")
    fom = (
        -np.log10(g2_0 + eps)
        + w_p * np.log10(purcell_factor + 1.0)
        + w_eta * extraction_efficiency
        - w_d * (dephasing_rate / target_dephasing) ** 2
    )
    return fom


def binary_encoded_parameter_search(
    n_bits_per_param: int,
    param_bounds: List[Tuple[float, float]],
    objective: Callable[[np.ndarray], float],
) -> Tuple[np.ndarray, float]:
    """
    使用二进制编码穷举所有参数组合（源自 satisfy_brute 的 2^N 搜索思想）。
    
    参数:
        n_bits_per_param: 每个参数的编码位数
        param_bounds: 每个参数的 (min, max) 范围
        objective: 目标函数，输入解码后的参数向量
    
    返回:
        best_params: 最优参数向量
        best_value: 最优目标值
    """
    n_params = len(param_bounds)
    total_bits = n_params * n_bits_per_param
    if total_bits > 20:
        raise ValueError("Search space too large (> 2^20 combinations)")
    n_combinations = 2 ** total_bits
    best_value = float('inf')
    best_params = None
    for idx in range(n_combinations):
        bvec = int_to_binary_vector(idx, total_bits)
        params = np.zeros(n_params, dtype=float)
        for p in range(n_params):
            bits = bvec[p * n_bits_per_param:(p + 1) * n_bits_per_param]
            # 二进制转整数，再映射到 [0, 1]
            int_val = 0
            for b in bits:
                int_val = int_val * 2 + int(b)
            frac = int_val / (2 ** n_bits_per_param - 1) if (2 ** n_bits_per_param - 1) > 0 else 0.0
            pmin, pmax = param_bounds[p]
            params[p] = pmin + frac * (pmax - pmin)
        val = objective(params)
        if val < best_value:
            best_value = val
            best_params = params.copy()
    return best_params, best_value


def sensitivity_analysis(
    base_params: Dict[str, float],
    param_deltas: Dict[str, float],
    objective: Callable[[Dict[str, float]], float],
) -> Dict[str, float]:
    """
    对参数进行一阶灵敏度分析：
    
        S_i = (f(x_i + delta_i) - f(x_i - delta_i)) / (2 delta_i)
    
    返回各参数的灵敏度系数。
    """
    base_val = objective(base_params)
    sensitivities = {}
    for name, delta in param_deltas.items():
        if abs(delta) < 1e-15:
            sensitivities[name] = 0.0
            continue
        p_plus = base_params.copy()
        p_minus = base_params.copy()
        p_plus[name] += delta
        p_minus[name] -= delta
        val_plus = objective(p_plus)
        val_minus = objective(p_minus)
        sens = (val_plus - val_minus) / (2.0 * delta)
        sensitivities[name] = sens
    return sensitivities
