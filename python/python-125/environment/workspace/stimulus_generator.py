"""
stimulus_generator.py
视觉刺激生成与神经连接组合探索

基于以下种子项目合成：
- 1273_toms515: 字典序组合生成
- 597_iplot: 字符串表达式求值
- 349_faure: Faure准随机序列

科学背景：
视觉系统处理多种类型的视觉刺激，包括：
1. 空间光栅（sinusoidal gratings）
2. 高斯blob
3. 运动刺激
4. 随机噪声（用于表征感受野）

本模块提供：
1. 标准视觉刺激生成
2. 神经连接子集的组合探索（基于toms515）
3. 参数化函数定义与求值
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 字典序组合生成（基于1273_toms515）
# =============================================================================

def binomial_coefficient(n: int, k: int) -> int:
    """
    计算二项式系数 C(n,k)。
    
    使用迭代乘法避免溢出：
        C(n,k) = Π_{i=1}^{k} (n-k+i) / i
    
    参数:
        n: 总数
        k: 选取数
    
    返回:
        C(n,k)
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    k = min(k, n - k)  # 利用对称性
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    
    return result


def comb_lexicographic(n: int, p: int, L: int) -> np.ndarray:
    """
    生成从n个元素中选取p个的第L个字典序组合。
    
    算法（ACM TOMS 515）：
    1. 初始化 c[0] = 0
    2. 对每个位置i从1到p：
       a. 设 c[i] = c[i-1] + 1
       b. 当 C(n-c[i], p-i) < L 时：
          L -= C(n-c[i], p-i)
          c[i] += 1
    3. 返回 c[1:]
    
    参数:
        n: 集合大小
        p: 组合大小
        L: 字典序索引（1-based）
    
    返回:
        combo: (p,) 组合数组（元素为1-based索引）
    """
    if L < 1 or L > binomial_coefficient(n, p):
        raise ValueError(f"L={L} out of range [1, C({n},{p})]")
    
    c = np.zeros(p + 1, dtype=np.int64)
    remaining = L
    
    for i in range(1, p + 1):
        c[i] = c[i - 1] + 1
        while True:
            bc = binomial_coefficient(n - c[i], p - i)
            if bc < remaining:
                remaining -= bc
                c[i] += 1
            else:
                break
    
    return c[1:]


def explore_synaptic_combinations(
    n_synapses: int,
    subset_size: int,
    n_combinations: int
) -> np.ndarray:
    """
    探索神经突触连接的不同子集组合。
    
    在视网膜中，一个神经节细胞可能接收来自数百个双极细胞的输入。
    通过组合生成，可以系统性地研究不同突触子集对响应的影响。
    
    参数:
        n_synapses: 总突触数量
        subset_size: 子集大小
        n_combinations: 要探索的组合数
    
    返回:
        combinations: (n_combinations, subset_size) 组合矩阵
    """
    total = binomial_coefficient(n_synapses, subset_size)
    n_combinations = min(n_combinations, total)
    
    # 均匀选择字典序索引
    indices = np.linspace(1, total, n_combinations, dtype=np.int64)
    
    combinations = np.zeros((n_combinations, subset_size), dtype=np.int64)
    for i, L in enumerate(indices):
        combinations[i] = comb_lexicographic(n_synapses, subset_size, int(L))
    
    return combinations


# =============================================================================
# 视觉刺激生成
# =============================================================================

def sinusoidal_grating(
    nx: int,
    ny: int,
    spatial_freq: float,
    orientation: float,
    contrast: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    生成正弦光栅视觉刺激。
    
    光栅方程：
        I(x,y) = contrast * sin(2π * f * (x*cosθ + y*sinθ) + φ)
    
    其中：
    - f: 空间频率 (cycles/degree)
    - θ: 朝向角 (radians)
    - φ: 相位 (radians)
    
    参数:
        nx, ny: 网格尺寸
        spatial_freq: 空间频率
        orientation: 朝向角（弧度）
        contrast: 对比度 [0,1]
        phase: 相位（弧度）
    
    返回:
        stimulus: (ny, nx) 刺激强度矩阵
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    
    # 旋转坐标
    X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
    
    stimulus = contrast * np.sin(2.0 * np.pi * spatial_freq * X_rot + phase)
    
    # 归一化到[0,1]
    stimulus = (stimulus + 1.0) / 2.0
    
    return stimulus


def gaussian_blob(
    nx: int,
    ny: int,
    sigma_x: float,
    sigma_y: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    生成二维高斯blob刺激。
    
    方程：
        I(x,y) = A * exp(-[(x-x₀)²/(2σ_x²) + (y-y₀)²/(2σ_y²)])
    
    参数:
        nx, ny: 网格尺寸
        sigma_x, sigma_y: x和y方向的标准差
        center_x, center_y: 中心位置（归一化到[-1,1]）
        amplitude: 振幅
    
    返回:
        stimulus: (ny, nx) 刺激强度矩阵
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    
    stimulus = amplitude * np.exp(
        -((X - center_x) ** 2 / (2.0 * sigma_x ** 2) +
          (Y - center_y) ** 2 / (2.0 * sigma_y ** 2))
    )
    
    return stimulus


def white_noise_stimulus(
    nx: int,
    ny: int,
    seed: int = 42
) -> np.ndarray:
    """
    生成白噪声刺激（用于反向相关感受野映射）。
    
    每个像素独立采样自均匀分布U[0,1]或高斯分布N(0.5, σ²)。
    
    参数:
        nx, ny: 网格尺寸
        seed: 随机种子
    
    返回:
        stimulus: (ny, nx) 噪声刺激矩阵
    """
    np.random.seed(seed)
    return np.random.random((ny, nx))


def drifting_grating(
    nx: int,
    ny: int,
    n_frames: int,
    spatial_freq: float,
    temporal_freq: float,
    orientation: float,
    dt: float = 0.01,
    contrast: float = 1.0
) -> np.ndarray:
    """
    生成漂移光栅刺激的时间序列。
    
    时间演化方程：
        I(x,y,t) = contrast * sin(2π * f_s * (x*cosθ + y*sinθ) - 2π * f_t * t + φ)
    
    参数:
        nx, ny: 空间网格尺寸
        n_frames: 帧数
        spatial_freq: 空间频率
        temporal_freq: 时间频率 (Hz)
        orientation: 朝向角（弧度）
        dt: 时间步长 (s)
        contrast: 对比度
    
    返回:
        stimulus_seq: (n_frames, ny, nx) 时间序列
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    
    X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
    
    stimulus_seq = np.zeros((n_frames, ny, nx), dtype=np.float64)
    for frame in range(n_frames):
        t = frame * dt
        phase = -2.0 * np.pi * temporal_freq * t
        stim = contrast * np.sin(2.0 * np.pi * spatial_freq * X_rot + phase)
        stimulus_seq[frame] = (stim + 1.0) / 2.0
    
    return stimulus_seq


# =============================================================================
# 函数表达式解析与求值（基于597_iplot）
# =============================================================================

def safe_math_eval(expr_str: str, x: float) -> float:
    """
    安全地求值数学表达式字符串。
    
    支持的函数：sin, cos, tan, exp, log, sqrt, pi, e
    
    参数:
        expr_str: 表达式字符串，使用'x'作为变量
        x: 变量值
    
    返回:
        result: 求值结果
    """
    safe_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }
    
    try:
        # 安全替换
        expr = expr_str.lower().replace('^', '**')
        result = eval(expr, {"__builtins__": {}}, {**safe_dict, 'x': x})
        return float(result)
    except Exception:
        return 0.0


def evaluate_tuning_function(
    func_str: str,
    x_values: np.ndarray
) -> np.ndarray:
    """
    对一组x值求值调谐函数字符串。
    
    参数:
        func_str: 函数表达式字符串
        x_values: (n,) 评估点
    
    返回:
        y_values: (n,) 函数值
    """
    return np.array([safe_math_eval(func_str, x) for x in x_values], dtype=np.float64)
