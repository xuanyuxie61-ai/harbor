"""
synaptic_diffusion.py
突触间隙神经递质反应-扩散模型

基于以下种子项目合成：
- 486_gray_scott_movie: Gray-Scott反应扩散系统模拟

科学背景：
视网膜中光感受器→双极细胞→神经节细胞的信号传递通过化学突触进行。
突触间隙中的神经递质（如谷氨酸）释放、扩散和回收过程可用反应-扩散方程描述。

Gray-Scott模型描述了两种化学物质U（神经递质前体）和V（活性神经递质）
在二维空间中的反应-扩散动力学：

    ∂U/∂t = D_u * ∇²U - U*V² + F*(1-U)
    ∂V/∂t = D_v * ∇²V + U*V² - (F+K)*V

其中：
- U: 未释放的神经递质囊泡密度（或前体浓度）
- V: 突触间隙中游离的神经递质浓度
- D_u, D_v: 扩散系数
- F: 神经递质补充速率（feed rate）
- K: 神经递质清除/重摄取速率（kill rate）
- ∇²: 二维Laplacian算子
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 9点Laplacian算子（基于486_gray_scott_movie）
# =============================================================================

def laplacian9_torus(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    在二维周期性边界（torus）网格上使用9点stencil计算Laplacian。
    
    9点stencil提供O(h⁴)精度：
    
        L = (1/(6*dx²)) * [ 1   4   1 ]
                            [ 4 -20   4 ]
                            [ 1   4   1 ]  * A
    
    即：
        L_{i,j} = (1/(6*dx²)) * [
            4*A_{i-1,j} + 4*A_{i+1,j} + 4*A_{i,j-1} + 4*A_{i,j+1}
            + A_{i-1,j-1} + A_{i-1,j+1} + A_{i+1,j-1} + A_{i+1,j+1}
            - 20*A_{i,j}
        ]
    
    周期性边界条件：网格首尾行列等价，A[-1,j] = A[-2,j], A[0,j] = A[1,j] 等
    （这里使用环形wrap-around）
    
    参数:
        field: (nx, ny) 标量场
        dx, dy: 空间步长
    
    返回:
        laplacian: (nx, ny) Laplacian结果
    """
    nx, ny = field.shape
    L = np.zeros_like(field)
    
    # 使用循环卷积实现周期性边界
    # 为了避免复杂的索引，使用numpy的roll
    for i in range(nx):
        for j in range(ny):
            # 周期性邻居索引
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            L[i, j] = (
                4.0 * field[im1, j] + 4.0 * field[ip1, j] +
                4.0 * field[i, jm1] + 4.0 * field[i, jp1] +
                field[im1, jm1] + field[im1, jp1] +
                field[ip1, jm1] + field[ip1, jp1] -
                20.0 * field[i, j]
            ) / (6.0 * dx * dy)
    
    return L


def laplacian5_zero_boundary(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    使用5点stencil计算Laplacian，零Dirichlet边界条件。
    
    5点stencil：
        L_{i,j} = (U_{i-1,j} + U_{i+1,j} + U_{i,j-1} + U_{i,j+1} - 4*U_{i,j}) / h²
    
    参数:
        field: (nx, ny) 标量场
        dx, dy: 空间步长
    
    返回:
        laplacian: (nx, ny) Laplacian结果
    """
    nx, ny = field.shape
    L = np.zeros_like(field)
    h2 = dx * dy
    
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            L[i, j] = (
                field[i - 1, j] + field[i + 1, j] +
                field[i, j - 1] + field[i, j + 1] -
                4.0 * field[i, j]
            ) / h2
    
    # 边界保持为0（Dirichlet边界）
    return L


# =============================================================================
# Gray-Scott反应-扩散求解器
# =============================================================================

def gray_scott_synaptic_step(
    U: np.ndarray,
    V: np.ndarray,
    Du: float,
    Dv: float,
    F: float,
    K: float,
    dt: float,
    dx: float,
    dy: float,
    boundary: str = 'periodic'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行一步Gray-Scott反应-扩散方程的时间推进。
    
    使用显式前向欧拉方法：
        U_{new} = U + dt * [D_u * ∇²U - U*V² + F*(1-U)]
        V_{new} = V + dt * [D_v * ∇²V + U*V² - (F+K)*V]
    
    稳定性条件（CFL-like）：
        dt ≤ dx² / (4 * max(D_u, D_v))
    
    参数:
        U, V: 当前神经递质浓度场
        Du, Dv: 扩散系数
        F: 补充速率
        K: 清除速率
        dt: 时间步长
        dx, dy: 空间步长
        boundary: 'periodic' 或 'zero'
    
    返回:
        U_new, V_new: 更新后的浓度场
    """
    if boundary == 'periodic':
        LU = laplacian9_torus(U, dx, dy)
        LV = laplacian9_torus(V, dx, dy)
    else:
        LU = laplacian5_zero_boundary(U, dx, dy)
        LV = laplacian5_zero_boundary(V, dx, dy)
    
    # 反应项
    UV2 = U * V ** 2
    
    # 前向欧拉更新
    U_new = U + dt * (Du * LU - UV2 + F * (1.0 - U))
    V_new = V + dt * (Dv * LV + UV2 - (F + K) * V)
    
    # 数值鲁棒性：截断到[0,1]
    U_new = np.clip(U_new, 0.0, 1.0)
    V_new = np.clip(V_new, 0.0, 1.0)
    
    return U_new, V_new


def simulate_synaptic_transmission(
    nx: int,
    ny: int,
    n_steps: int,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.035,
    K: float = 0.060,
    dt: float = 1.0,
    dx: float = 0.5,
    dy: float = 0.5,
    initial_condition: str = 'localized',
    boundary: str = 'periodic'
) -> dict:
    """
    模拟突触间隙中神经递质的时空演化。
    
    初始条件选项：
    - 'localized': 中心局部高浓度（模拟单个突触小泡释放）
    - 'wavefront': 平面波前（模拟扩散波）
    - 'random': 随机分布
    
    参数:
        nx, ny: 网格尺寸
        n_steps: 时间步数
        Du, Dv: 扩散系数
        F: 补充速率
        K: 清除速率
        dt: 时间步长
        dx, dy: 空间步长
        initial_condition: 初始条件类型
        boundary: 边界条件类型
    
    返回:
        result: 包含U_history, V_history, final_U, final_V的字典
    """
    # 稳定性检查
    stable_dt = (dx * dy) / (4.0 * max(Du, Dv) + 1e-14)
    if dt > stable_dt:
        print(f"Warning: dt={dt} exceeds stability limit {stable_dt:.4f}. Adjusting...")
        dt = 0.5 * stable_dt
    
    # 初始化
    U = np.ones((nx, ny), dtype=np.float64)
    V = np.zeros((nx, ny), dtype=np.float64)
    
    if initial_condition == 'localized':
        # 中心局部高浓度V（突触小泡释放位点）
        cx, cy = nx // 2, ny // 2
        radius = min(nx, ny) // 10
        for i in range(nx):
            for j in range(ny):
                dist2 = (i - cx) ** 2 + (j - cy) ** 2
                if dist2 < radius ** 2:
                    U[i, j] = 0.5
                    V[i, j] = 0.25
    elif initial_condition == 'wavefront':
        # 左半平面高V
        for i in range(nx):
            for j in range(ny):
                if i < nx // 4:
                    U[i, j] = 0.5
                    V[i, j] = 0.25
    elif initial_condition == 'random':
        np.random.seed(42)
        noise = np.random.random((nx, ny))
        U = U - 0.1 * noise
        V = 0.1 * noise
    
    # 存储历史
    save_interval = max(1, n_steps // 100)
    n_saved = n_steps // save_interval + 1
    U_history = np.zeros((n_saved, nx, ny), dtype=np.float64)
    V_history = np.zeros((n_saved, nx, ny), dtype=np.float64)
    
    U_history[0] = U.copy()
    V_history[0] = V.copy()
    save_idx = 1
    
    # 时间推进
    for step in range(n_steps):
        U, V = gray_scott_synaptic_step(U, V, Du, Dv, F, K, dt, dx, dy, boundary)
        
        if (step + 1) % save_interval == 0 and save_idx < n_saved:
            U_history[save_idx] = U.copy()
            V_history[save_idx] = V.copy()
            save_idx += 1
    
    return {
        'U_history': U_history[:save_idx],
        'V_history': V_history[:save_idx],
        'final_U': U,
        'final_V': V,
        'n_steps': n_steps,
        'dt': dt,
        'save_interval': save_interval,
    }


# =============================================================================
# 突触传递效能计算
# =============================================================================

def compute_synaptic_efficacy(
    V_field: np.ndarray,
    threshold: float = 0.1,
    receptor_density: np.ndarray = None
) -> dict:
    """
    计算突触传递的效能指标。
    
    突触效能取决于：
    1. 突触间隙中神经递质浓度V的时空分布
    2. 突触后膜受体密度
    
    突触后电流近似为：
        I_syn = g_max * [V] / ([V] + K_d) * (V_post - E_rev)
    
    其中K_d为解离常数，g_max为最大电导。
    
    参数:
        V_field: (nx, ny) 神经递质浓度场
        threshold: 有效传递浓度阈值
        receptor_density: (nx, ny) 受体密度分布（若None则均匀）
    
    返回:
        metrics: 包含peak_concentration, active_area, total_efficacy的字典
    """
    if receptor_density is None:
        receptor_density = np.ones_like(V_field)
    
    peak_conc = float(np.max(V_field))
    mean_conc = float(np.mean(V_field))
    active_mask = V_field > threshold
    active_area = int(np.sum(active_mask))
    
    # 加权效能
    efficacy_field = V_field * receptor_density
    total_efficacy = float(np.sum(efficacy_field))
    
    return {
        'peak_concentration': peak_conc,
        'mean_concentration': mean_conc,
        'active_area_pixels': active_area,
        'total_efficacy': total_efficacy,
        'active_fraction': active_area / V_field.size,
    }
