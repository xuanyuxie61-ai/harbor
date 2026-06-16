"""
参数扫描模块 (from ode_sweep_parfor)

实现 (k, omega) 空间的并行化参数扫描, 用于:
- 数值色散关系全面扫描
- 稳定性图构建
- 收敛性测试

融合 1029_Fdl1989 的矩阵生成思想与 832_ode_sweep 的参数扫描模式
"""

import numpy as np
from typing import Callable, Tuple, List, Dict
try:
    from plasma_constants import (plasma_frequency_electron, thermal_velocity,
                                  debye_length, ELECTRON_MASS)
except ImportError:
    from plasma_constants import (plasma_frequency_electron, thermal_velocity,
                                  debye_length, ELECTRON_MASS)
try:
    from dispersion_analysis import (numerical_dispersion_sweep,
                                     phase_velocity_error, group_velocity)
except ImportError:
    from dispersion_analysis import (numerical_dispersion_sweep,
                                     phase_velocity_error, group_velocity)


def parameter_sweep_1d(param_range: np.ndarray,
                          func: Callable[[float], float]) -> np.ndarray:
    """
    1D 参数扫描:
        results[i] = func(param_range[i])
    """
    results = np.zeros(len(param_range))
    for i, p in enumerate(param_range):
        results[i] = func(p)
    return results


def parameter_sweep_2d(param1_range: np.ndarray,
                          param2_range: np.ndarray,
                          func: Callable[[float, float], float]) -> np.ndarray:
    """
    2D 参数扫描:
        results[i, j] = func(param1_range[i], param2_range[j])
    """
    n1 = len(param1_range)
    n2 = len(param2_range)
    results = np.zeros((n1, n2))
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            results[i, j] = func(p1, p2)
    return results


def dispersion_sweep(k_array: np.ndarray, omega_pe: float,
                        v_th: float, dx_values: np.ndarray,
                        dt_values: np.ndarray,
                        fd_orders: List[int]) -> Dict:
    """
    全面色散扫描: (k, dx, dt, fd_order) -> omega

    Returns:
        dict with 'k', 'dx', 'dt', 'fd_order', 'omega' arrays
    """
    results = {}
    for fd_order in fd_orders:
        for dx in dx_values:
            for dt in dt_values:
                omega = numerical_dispersion_sweep(k_array, omega_pe, v_th,
                                                      dx, dt, fd_order)
                key = (fd_order, float(dx), float(dt))
                results[key] = omega

    return results


def stability_diagram(dx_array: np.ndarray, dt_array: np.ndarray,
                         omega_pe: float, v_th: float,
                         fd_order: int = 2,
                         k_test: float = None) -> np.ndarray:
    """
    稳定性图: (dx, dt) 空间

    对每个 (dx, dt) 点检查数值色散是否有虚部 (不稳定)

    Returns:
        stability_map : [len(dx), len(dt)] 数组
                         0 = 稳定, 1 = 不稳定, NaN = 未定义
    """
    from dispersion_analysis import numerical_dispersion_relation

    if k_test is None:
        k_test = 0.5 * np.pi / np.min(dx_array)

    stability_map = np.zeros((len(dx_array), len(dt_array)))

    for i, dx in enumerate(dx_array):
        for j, dt in enumerate(dt_array):
            try:
                omega = numerical_dispersion_relation(k_test, omega_pe, v_th,
                                                         dx, dt, fd_order)
                if np.imag(omega) > 1e-6:
                    stability_map[i, j] = 1.0  # 不稳定
                elif np.isnan(np.real(omega)):
                    stability_map[i, j] = np.nan
                else:
                    stability_map[i, j] = 0.0  # 稳定
            except (ValueError, ZeroDivisionError):
                stability_map[i, j] = np.nan

    return stability_map


def convergence_study(N_particles: np.ndarray, N_grid: int,
                        L: float, n_steps: int,
                        run_pic_func: Callable,
                        quantity_func: Callable) -> Dict:
    """
    粒子数收敛性研究:
        对不同 N_particles 运行 PIC, 记录指定物理量

    Parameters:
        N_particles   : 粒子数数组
        N_grid        : 网格点数
        L             : 域长度
        n_steps       : 时间步数
        run_pic_func  : PIC 运行函数
        quantity_func : 从结果提取物理量的函数

    Returns:
        dict with 'Np', 'quantity', 'error' (if reference available)
    """
    results = {
        'Np': N_particles.copy(),
        'quantity': np.zeros(len(N_particles))
    }

    for i, Np in enumerate(N_particles):
        pic_result = run_pic_func(int(Np), N_grid, L, n_steps)
        results['quantity'][i] = quantity_func(pic_result)

    # 误差 (相对于最细粒子数)
    ref = results['quantity'][-1]
    if abs(ref) > 1e-300:
        results['error'] = np.abs(results['quantity'] - ref) / abs(ref)
    else:
        results['error'] = np.abs(results['quantity'] - ref)

    return results


def grid_convergence_study(N_grid_array: np.ndarray, N_particles: int,
                              L: float, n_steps: int,
                              run_pic_func: Callable,
                              quantity_func: Callable,
                              fd_orders: List[int]) -> Dict:
    """
    网格收敛性研究 (对多个 fd_order)
    """
    results = {}
    for fd_order in fd_orders:
        results[fd_order] = {
            'N_grid': N_grid_array.copy(),
            'quantity': np.zeros(len(N_grid_array))
        }
        for i, Ng in enumerate(N_grid_array):
            pic_result = run_pic_func(N_particles, int(Ng), L, n_steps,
                                        fd_order=fd_order)
            results[fd_order]['quantity'][i] = quantity_func(pic_result)

    return results


def matrix_generation_sweep(rows: int, cols: int,
                              param_func: Callable) -> np.ndarray:
    """
    矩阵生成扫描 (from 1029_Fdl1989):
        M[i, j] = param_func(i, j)
    """
    M = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            M[i, j] = param_func(i, j)
    return M


def sweep_with_timing(param_range: np.ndarray,
                         func: Callable) -> Dict:
    """
    带计时的参数扫描 (性能分析)
    """
    import time
    times = np.zeros(len(param_range))
    results = np.zeros(len(param_range))

    for i, p in enumerate(param_range):
        t0 = time.perf_counter()
        results[i] = func(p)
        t1 = time.perf_counter()
        times[i] = t1 - t0

    return {
        'param': param_range,
        'result': results,
        'time': times,
        'time_per_call': times
    }


def generate_k_array(k_min: float, k_max: float, n_k: int,
                       mode: str = 'linear') -> np.ndarray:
    """
    生成 k 值数组

    mode = 'linear'    : 线性
    mode = 'log'       : 对数
    mode = 'debye'     : 以 k*lambda_D 为基准
    """
    if mode == 'linear':
        return np.linspace(k_min, k_max, n_k)
    elif mode == 'log':
        return np.logspace(np.log10(max(k_min, 1e-10)), np.log10(k_max), n_k)
    elif mode == 'debye':
        return np.linspace(k_min, k_max, n_k)
    else:
        raise ValueError(f"未知模式: {mode}")


def batch_dispersion_analysis(omega_pe: float, v_th: float,
                                  dx: float, dt: float,
                                  n_k: int = 50,
                                  fd_orders: List[int] = None) -> Dict:
    """
    批量色散分析

    Returns:
        dict with 'k', 'omega' for each fd_order
    """
    if fd_orders is None:
        fd_orders = [2, 4, 6]

    k_max = np.pi / dx * 0.9  # 避免 Nyquist 边界
    k_array = np.linspace(1e-3, k_max, n_k)

    results = {'k': k_array}
    for fd_order in fd_orders:
        omega = numerical_dispersion_sweep(k_array, omega_pe, v_th,
                                              dx, dt, fd_order)
        results[f'omega_fd{fd_order}'] = omega

    return results
