"""
气压场梯度增强与空间滤波模块
==============================
基于种子项目 574_image_contrast 的邻域对比度增强思想。

核心科学问题：
    台风结构分析需要清晰识别气压场中的梯度区（眼墙、外围锋面等）。
    本模块将图像对比度增强技术推广至球面气象场，实现：
    
    1. 空间梯度增强（类似于图像锐化）
    2. 各向异性扩散平滑（Anisotropic Diffusion）
    3. 边界层锋面检测

数学模型：

=== 1. 梯度增强滤波 ===

基于 574_image_contrast_gray 的邻域平均思想：
    对于场变量 Φ，在网格点 i 处的局部平均：
        Φ̄_i = (1/|N(i)|) Σ_{j∈N(i)} Φ_j
    
    增强后的场：
        Φ^{enh}_i = s * Φ_i + (1-s) * Φ̄_i
    
    其中 s > 1 为锐化因子。

=== 2. 各向异性扩散滤波 ===

Perona-Malik 模型：
    ∂Φ/∂t = div( c(|∇Φ|) * ∇Φ )
    
    其中扩散系数：
        c(s) = 1 / (1 + (s/λ)²)
    
    或
        c(s) = exp(-(s/λ)²)
    
    λ 为对比度参数，控制边缘保留强度。

离散形式（显式）：
    Φ^{n+1}_i = Φ^n_i + Δt * Σ_{j∈N(i)} c(|Φ^n_j - Φ^n_i|) * (Φ^n_j - Φ^n_i) / d_{ij}²

=== 3. 锋面检测 ===

使用梯度幅值和拉普拉斯零交叉：
    |∇Φ| > threshold 且 ∇²Φ 变号
"""

import numpy as np


def local_average_1d(field, boundary='symmetric'):
    """
    计算一维场的局部邻域平均（3点 stencil）。
    
    基于 574_image_contrast_gray 的邻域平均思想。
    
    参数:
        field: 一维场
        boundary: 边界处理方式
    
    返回:
        avg: 局部平均场
    """
    n = len(field)
    avg = np.zeros(n)
    
    # 内部点
    avg[1:-1] = (field[:-2] + field[1:-1] + field[2:]) / 3.0
    
    # 边界
    if boundary == 'symmetric':
        avg[0] = (2.0 * field[0] + field[1]) / 3.0
        avg[-1] = (2.0 * field[-1] + field[-2]) / 3.0
    elif boundary == 'periodic':
        avg[0] = (field[-1] + field[0] + field[1]) / 3.0
        avg[-1] = (field[-2] + field[-1] + field[0]) / 3.0
    else:
        avg[0] = field[0]
        avg[-1] = field[-1]
    
    return avg


def gradient_enhancement_1d(field, sharpness=1.5, boundary='symmetric'):
    """
    一维场梯度增强（锐化）。
    
    公式：
        Φ^{enh} = s * Φ + (1-s) * Φ̄
    
    参数:
        field: 输入场
        sharpness: 锐化因子 s，s > 1 增强梯度，0 < s < 1 平滑
        boundary: 边界处理
    
    返回:
        enhanced: 增强后的场
    """
    avg = local_average_1d(field, boundary)
    enhanced = sharpness * field + (1.0 - sharpness) * avg
    return enhanced


def compute_gradient_1d(x, field):
    """
    计算一维场的空间梯度（中心差分）。
    
    公式：
        dΦ/dx|_i ≈ (Φ_{i+1} - Φ_{i-1}) / (x_{i+1} - x_{i-1})
    
    参数:
        x: 坐标网格
        field: 场值
    
    返回:
        gradient: 梯度
    """
    n = len(field)
    grad = np.zeros(n)
    
    dx_forward = np.diff(x)
    
    # 内部点
    grad[1:-1] = (field[2:] - field[:-2]) / (x[2:] - x[:-2])
    
    # 边界：一阶差分
    if n > 1:
        grad[0] = (field[1] - field[0]) / dx_forward[0]
        grad[-1] = (field[-1] - field[-2]) / dx_forward[-1]
    
    return grad


def compute_laplacian_1d(x, field):
    """
    计算一维场的 Laplacian（二阶中心差分）。
    
    公式（非均匀网格）：
        ∇²Φ|_i ≈ 2 * [ (Φ_{i+1} - Φ_i)/(x_{i+1} - x_i) - (Φ_i - Φ_{i-1})/(x_i - x_{i-1}) ]
                    / (x_{i+1} - x_{i-1})
    
    参数:
        x: 坐标网格
        field: 场值
    
    返回:
        laplacian: Laplacian
    """
    n = len(field)
    lap = np.zeros(n)
    
    dx = np.diff(x)
    
    for i in range(1, n - 1):
        dx_forward = x[i + 1] - x[i]
        dx_backward = x[i] - x[i - 1]
        dx_total = x[i + 1] - x[i - 1]
        
        if dx_forward > 0 and dx_backward > 0 and dx_total > 0:
            lap[i] = 2.0 * ((field[i + 1] - field[i]) / dx_forward
                            - (field[i] - field[i - 1]) / dx_backward) / dx_total
    
    return lap


def anisotropic_diffusion_1d(field, x, n_iter=10, dt=0.1, lambda_param=1.0,
                              diffusion_type='exponential'):
    """
    一维各向异性扩散滤波（Perona-Malik）。
    
    参数:
        field: 输入场
        x: 坐标网格
        n_iter: 迭代次数
        dt: 伪时间步长
        lambda_param: 对比度参数
        diffusion_type: 'exponential' 或 'fractional'
    
    返回:
        filtered: 滤波后的场
    """
    phi = field.copy()
    n = len(phi)
    
    for _ in range(n_iter):
        phi_new = phi.copy()
        
        for i in range(1, n - 1):
            dx_forward = x[i + 1] - x[i]
            dx_backward = x[i] - x[i - 1]
            
            if dx_forward <= 0 or dx_backward <= 0:
                continue
            
            # 前向梯度
            grad_forward = (phi[i + 1] - phi[i]) / dx_forward
            # 后向梯度
            grad_backward = (phi[i] - phi[i - 1]) / dx_backward
            
            # 扩散系数
            if diffusion_type == 'exponential':
                c_forward = np.exp(-(abs(grad_forward) / lambda_param)**2)
                c_backward = np.exp(-(abs(grad_backward) / lambda_param)**2)
            else:
                c_forward = 1.0 / (1.0 + (grad_forward / lambda_param)**2)
                c_backward = 1.0 / (1.0 + (grad_backward / lambda_param)**2)
            
            # 离散扩散项
            flux = c_forward * grad_forward - c_backward * grad_backward
            # 平均网格间距
            dx_avg = 0.5 * (dx_forward + dx_backward)
            phi_new[i] += dt * flux / dx_avg
        
        phi = phi_new
    
    return phi


def detect_fronts_1d(x, field, gradient_threshold=0.5, laplacian_threshold=0.1):
    """
    检测气压场中的锋面位置。
    
    锋面判据：
        1. |∇Φ| > gradient_threshold
        2. Laplacian 变号（零交叉）
    
    参数:
        x: 坐标
        field: 场值
        gradient_threshold: 梯度阈值
        laplacian_threshold: Laplacian 阈值
    
    返回:
        front_indices: 锋面位置索引列表
        front_strength: 锋面强度列表
    """
    grad = compute_gradient_1d(x, field)
    lap = compute_laplacian_1d(x, field)
    
    front_indices = []
    front_strength = []
    
    n = len(field)
    for i in range(1, n - 1):
        # 梯度阈值
        if abs(grad[i]) < gradient_threshold:
            continue
        
        # Laplacian 零交叉
        if lap[i - 1] * lap[i + 1] < 0 or abs(lap[i]) < laplacian_threshold:
            front_indices.append(i)
            front_strength.append(abs(grad[i]))
    
    return front_indices, front_strength


def apply_spatial_filter_pipeline(field, x, enhance=True, smooth=True, detect=True):
    """
    完整的空间滤波处理流程。
    
    参数:
        field: 输入气压场
        x: 坐标网格
        enhance: 是否进行梯度增强
        smooth: 是否进行各向异性平滑
        detect: 是否检测锋面
    
    返回:
        result: 处理后的场
        info: 处理信息字典
    """
    result = field.copy()
    info = {}
    
    if enhance:
        result = gradient_enhancement_1d(result, sharpness=1.3)
        info['enhanced'] = True
    
    if smooth:
        result = anisotropic_diffusion_1d(result, x, n_iter=5, lambda_param=2.0)
        info['smoothed'] = True
    
    if detect:
        fronts, strength = detect_fronts_1d(x, result)
        info['fronts'] = fronts
        info['front_strength'] = strength
        info['n_fronts'] = len(fronts)
    
    return result, info
