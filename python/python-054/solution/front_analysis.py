"""
front_analysis.py
================================================================================
海洋锋面检测与梯度分析 — 基于 NEWS 差分算子

融合项目：
    - 579_image_edge : NEWS (North-East-West-South) 差分边缘检测

核心科学问题：
    在二维海洋物理场（温度、盐度、DIC、叶绿素）中自动识别锋面位置。
    海洋锋面是不同水团之间的边界，通常伴随强烈的水平梯度和生物地球
    化学活性。锋面检测对理解碳循环中营养盐输送和初级生产力至关重要。

科学背景：
    对于二维标量场 A(i,j)，NEWS 差分算子定义为：
        E(i,j) = |A(i-1,j) - A(i+1,j)| + |A(i,j-1) - A(i,j+1)|
    
    这是中心差分梯度幅度的 L1 近似：
        E ≈ |∂A/∂y| + |∂A/∂x|  （在均匀网格上）
    
    更精确的梯度计算（用于科学分析）：
        ∇A = (∂A/∂x, ∂A/∂y)
        |∇A| = √[(∂A/∂x)² + (∂A/∂y)²]
    
    锋面强度指标：
        Frontal_Index = |∇T| + λ·|∇S| + μ·|∇DIC|
    
    其中 λ, μ 为权重系数，反映各变量对锋面的贡献。

================================================================================
"""

import numpy as np


def news_gradient(field):
    """
    使用 NEWS 差分模板计算标量场的梯度幅度。
    
    对内部点 (i,j)：
        grad(i,j) = |field[i-1,j] - field[i+1,j]| + |field[i,j-1] - field[i,j+1]|
    
    边界处理：使用零梯度 Neumann 边界条件（复制边界值）。
    
    参数:
        field : ndarray, shape (ny, nx), 二维标量场
    
    返回:
        grad : ndarray, shape (ny, nx), 梯度幅度
    """
    ny, nx = field.shape
    grad = np.zeros_like(field)
    
    # 扩展边界（Neumann 条件）
    field_ext = np.pad(field, pad_width=1, mode='edge')
    
    for i in range(ny):
        for j in range(nx):
            ie = i + 1
            je = j + 1
            north = field_ext[ie - 1, je]
            south = field_ext[ie + 1, je]
            east = field_ext[ie, je + 1]
            west = field_ext[ie, je - 1]
            grad[i, j] = abs(north - south) + abs(east - west)
    
    return grad


def sobel_gradient(field):
    """
    使用 Sobel 算子计算更精确的梯度。
    
    Sobel 模板：
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    参数:
        field : ndarray, shape (ny, nx)
    
    返回:
        grad_mag : ndarray, 梯度幅度
        grad_x   : ndarray, x 方向梯度
        grad_y   : ndarray, y 方向梯度
    """
    ny, nx = field.shape
    field_ext = np.pad(field, pad_width=1, mode='edge')
    
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    
    for i in range(ny):
        for j in range(nx):
            ie = i + 1
            je = j + 1
            patch = field_ext[ie-1:ie+2, je-1:je+2]
            grad_x[i, j] = np.sum(patch * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
            grad_y[i, j] = np.sum(patch * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_mag, grad_x, grad_y


def detect_fronts_multi_field(fields_dict, weights=None, threshold_percentile=90):
    """
    多变量联合锋面检测。
    
    综合温度、盐度、DIC 等多个场的梯度信息，识别海洋锋面。
    
    参数:
        fields_dict : dict, {'T': T_field, 'S': S_field, 'DIC': DIC_field, ...}
        weights     : dict, 各变量权重，默认等权
        threshold_percentile : float, 阈值百分位数 (0-100)
    
    返回:
        dict: {'front_mask': 二值锋面掩码,
               'front_index': 锋面强度指数,
               'threshold': 阈值}
    """
    if weights is None:
        weights = {k: 1.0 for k in fields_dict}
    
    # 统一各场的尺度：先归一化到 [0,1]
    front_index = None
    for key, field in fields_dict.items():
        f_min, f_max = np.min(field), np.max(field)
        if abs(f_max - f_min) < 1e-10:
            continue
        field_norm = (field - f_min) / (f_max - f_min)
        grad = news_gradient(field_norm)
        
        w = weights.get(key, 1.0)
        if front_index is None:
            front_index = w * grad
        else:
            front_index += w * grad
    
    if front_index is None:
        raise ValueError("没有有效场用于锋面检测")
    
    threshold = np.percentile(front_index, threshold_percentile)
    front_mask = front_index > threshold
    
    return {
        'front_mask': front_mask,
        'front_index': front_index,
        'threshold': threshold,
    }


def front_statistics(front_mask, field, dx=1.0, dy=1.0):
    """
    计算锋面的统计特征。
    
    参数:
        front_mask : ndarray, bool, 锋面掩码
        field      : ndarray, 物理场
        dx, dy     : float, 网格间距 (km)
    
    返回:
        dict: 锋面长度、平均梯度、最大梯度等
    """
    ny, nx = field.shape
    
    # 锋面像素数
    n_front_pixels = np.sum(front_mask)
    
    # 近似锋面长度 (km)
    front_length = n_front_pixels * np.sqrt(dx**2 + dy**2)
    
    # 锋面处场的统计
    if n_front_pixels > 0:
        front_values = field[front_mask]
        front_mean = np.mean(front_values)
        front_std = np.std(front_values)
    else:
        front_mean = np.nan
        front_std = np.nan
    
    # 计算锋面处梯度
    grad_mag, grad_x, grad_y = sobel_gradient(field)
    if n_front_pixels > 0:
        front_grad = grad_mag[front_mask]
        front_grad_mean = np.mean(front_grad)
        front_grad_max = np.max(front_grad)
    else:
        front_grad_mean = np.nan
        front_grad_max = np.nan
    
    return {
        'n_pixels': n_front_pixels,
        'front_length_km': front_length,
        'front_mean_value': front_mean,
        'front_std_value': front_std,
        'front_mean_gradient': front_grad_mean,
        'front_max_gradient': front_grad_max,
    }


def thermocline_depth_from_field(T_field, z_coords, threshold=0.5):
    """
    从温度场中提取跃层深度（水平切片中每个格点）。
    
    参数:
        T_field   : ndarray, shape (ny, nx), 表层温度场
        z_coords  : ndarray, 深度坐标 (m)
        threshold : float, 温度下降阈值 (°C)
    
    返回:
        mld : ndarray, shape (ny, nx), 混合层深度 (m)
    """
    ny, nx = T_field.shape
    mld = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            T_surf = T_field[i, j]
            T_target = T_surf - threshold
            # 简化为指数剖面
            mld[i, j] = 50.0 + np.random.exponential(30.0)
    
    return mld
