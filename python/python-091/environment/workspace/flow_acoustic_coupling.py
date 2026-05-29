"""
流-声耦合背景场建模与FEM插值模块

基于种子项目 787_navier_stokes_2d_exact 和 425_ffmatlib 的核心算法，
为超声层析成像提供非均匀流场背景下的声波传播模拟。

物理模型:
当声波在非均匀流动介质中传播时，声波方程修正为:
    (∂/∂t + v·∇)²p - c₀²∇²p = 0
其中 v(x,y) 为背景流速场。

在时谐假设下（p = P·exp(-iωt)），得到对流Helmholtz方程:
    (iω + v·∇)²P + ω²/c₀²·P + c₀²∇²P = 0

对于低速流动（Mach数 Ma = |v|/c₀ ≪ 1），可简化为:
    ∇²P + k²·P = -2ik/c₀ · (v·∇P)
    （右侧为流-声耦合源项）

Navier-Stokes精确解:
本项目采用Taylor-Green涡作为背景流场，其解析表达式为:
    u(x,y,t) =  sin(x)·cos(y)·exp(-2νt)
    v(x,y,t) = -cos(x)·sin(y)·exp(-2νt)
    p(x,y,t) =  0.25·(cos(2x) + cos(2y))·exp(-4νt)
"""

import numpy as np
from typing import Tuple, Callable


def taylor_green_vortex(x: np.ndarray, y: np.ndarray, t: float = 0.0,
                        nu: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Taylor-Green涡的精确解析解。
    
    这是Navier-Stokes方程在周期边界条件下的经典解析解，
    常用于验证CFD求解器的精度。
    
    公式:
        u(x,y,t) =  sin(x)·cos(y)·exp(-2νt)
        v(x,y,t) = -cos(x)·sin(y)·exp(-2νt)
        p(x,y,t) = 0.25·(cos(2x) + cos(2y))·exp(-4νt)
    
    参数:
        x, y: 空间坐标网格（2D数组或标量）
        t: 时间
        nu: 运动粘滞系数
    
    返回:
        u, v: 速度分量 (m/s)
        p: 压力场 (Pa)
    """
    decay_v = np.exp(-2.0 * nu * t)
    decay_p = np.exp(-4.0 * nu * t)
    
    u = np.sin(x) * np.cos(y) * decay_v
    v = -np.cos(x) * np.sin(y) * decay_v
    p = 0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay_p
    
    return u, v, p


def cavity_flow_exact(x: np.ndarray, y: np.ndarray, Re: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """二维顶盖驱动腔体流的近似解析解（Ghia et al. 多项式拟合）。
    
    用于模拟血管或组织间隙中的缓慢流动。
    
    参数:
        x, y: 空间坐标（归一化到 [0,1]×[0,1]）
        Re: Reynolds数
    
    返回:
        u, v: 速度分量
    """
    # 边界检查
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    
    # 简化的多项式近似
    u = 16.0 * x**2 * (1.0 - x)**2 * y * (1.0 - y) * (2.0 * y - 1.0) * Re / 100.0
    v = -16.0 * x * (1.0 - x) * (2.0 * x - 1.0) * y**2 * (1.0 - y)**2 * Re / 100.0
    
    return u, v


def interpolate_to_grid(tri_nodes: np.ndarray, tri_elements: np.ndarray,
                        node_values: np.ndarray,
                        grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """将三角形网格上的P1有限元数据插值到规则矩形网格。
    
    P1有限元基函数: φᵢ 在节点i处为1，在其他节点处为0，
    在三角形内部为线性函数。
    
    对于点 P = (x,y) 落在三角形 T = {(x₁,y₁), (x₂,y₂), (x₃,y₃)} 内:
        P = λ₁·P₁ + λ₂·P₂ + λ₃·P₃
    其中 λᵢ 为面积坐标（重心坐标），满足 λ₁+λ₂+λ₃=1。
    
    插值公式:
        u(P) = λ₁·u₁ + λ₂·u₂ + λ₃·u₃
    
    参数:
        tri_nodes: (N, 2) 三角形网格节点坐标
        tri_elements: (M, 3) 三角形单元索引
        node_values: (N,) 节点上的标量值
        grid_x: (nx,) x方向规则网格坐标
        grid_y: (ny,) y方向规则网格坐标
    
    返回:
        grid_values: (ny, nx) 规则网格上的插值结果
    """
    nx = len(grid_x)
    ny = len(grid_y)
    grid_values = np.zeros((ny, nx))
    
    # 构建三角形包围盒索引以加速查询
    n_tri = tri_elements.shape[0]
    tri_min_x = np.zeros(n_tri)
    tri_max_x = np.zeros(n_tri)
    tri_min_y = np.zeros(n_tri)
    tri_max_y = np.zeros(n_tri)
    
    for t in range(n_tri):
        idx = tri_elements[t]
        pts = tri_nodes[idx]
        tri_min_x[t] = np.min(pts[:, 0])
        tri_max_x[t] = np.max(pts[:, 0])
        tri_min_y[t] = np.min(pts[:, 1])
        tri_max_y[t] = np.max(pts[:, 1])
    
    # 对每个规则网格点寻找所在三角形并插值
    for j in range(ny):
        for i in range(nx):
            px = grid_x[i]
            py = grid_y[j]
            
            found = False
            for t in range(n_tri):
                # 快速包围盒测试
                if px < tri_min_x[t] or px > tri_max_x[t]:
                    continue
                if py < tri_min_y[t] or py > tri_max_y[t]:
                    continue
                
                # 计算重心坐标
                idx = tri_elements[t]
                p1, p2, p3 = tri_nodes[idx[0]], tri_nodes[idx[1]], tri_nodes[idx[2]]
                
                denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
                if abs(denom) < 1e-14:
                    continue
                
                lam1 = ((p2[1] - p3[1]) * (px - p3[0]) + (p3[0] - p2[0]) * (py - p3[1])) / denom
                lam2 = ((p3[1] - p1[1]) * (px - p3[0]) + (p1[0] - p3[0]) * (py - p3[1])) / denom
                lam3 = 1.0 - lam1 - lam2
                
                # 检查是否在三角形内（含边界）
                if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
                    vals = node_values[idx]
                    grid_values[j, i] = lam1 * vals[0] + lam2 * vals[1] + lam3 * vals[2]
                    found = True
                    break
            
            if not found:
                # 最近邻插值作为fallback
                dists = np.linalg.norm(tri_nodes - np.array([px, py]), axis=1)
                nearest = np.argmin(dists)
                grid_values[j, i] = node_values[nearest]
    
    return grid_values


def compute_flow_acoustic_source(flow_u: np.ndarray, flow_v: np.ndarray,
                                 acoustic_p: np.ndarray,
                                 dx: float, dy: float,
                                 k: float, c0: float = 1540.0) -> np.ndarray:
    """计算流-声耦合源项。
    
    在Ma ≪ 1近似下，源项为:
        S = -2ik/c₀ · (v·∇P)
    
    其中 ∇P = (∂P/∂x, ∂P/∂y) 用中心差分近似:
        ∂P/∂x ≈ (P_{i+1,j} - P_{i-1,j}) / (2dx)
        ∂P/∂y ≈ (P_{i,j+1} - P_{i,j-1}) / (2dy)
    
    参数:
        flow_u, flow_v: 背景流速场 (ny, nx)
        acoustic_p: 声压场 (ny, nx)，复数
        dx, dy: 空间步长
        k: 声波波数
        c0: 背景声速
    
    返回:
        source: 耦合源项 (ny, nx)
    """
    ny, nx = acoustic_p.shape
    source = np.zeros_like(acoustic_p, dtype=complex)
    
    # 边界处理：使用Neumann边界条件（零梯度）
    for j in range(ny):
        for i in range(nx):
            # x方向导数
            if i == 0:
                dpx = (acoustic_p[j, i + 1] - acoustic_p[j, i]) / dx
            elif i == nx - 1:
                dpx = (acoustic_p[j, i] - acoustic_p[j, i - 1]) / dx
            else:
                dpx = (acoustic_p[j, i + 1] - acoustic_p[j, i - 1]) / (2.0 * dx)
            
            # y方向导数
            if j == 0:
                dpy = (acoustic_p[j + 1, i] - acoustic_p[j, i]) / dy
            elif j == ny - 1:
                dpy = (acoustic_p[j, i] - acoustic_p[j - 1, i]) / dy
            else:
                dpy = (acoustic_p[j + 1, i] - acoustic_p[j - 1, i]) / (2.0 * dy)
            
            # 流-声耦合项
            v_dot_grad_p = flow_u[j, i] * dpx + flow_v[j, i] * dpy
            source[j, i] = -2.0j * k / c0 * v_dot_grad_p
    
    return source


def mach_number_field(flow_u: np.ndarray, flow_v: np.ndarray,
                      c0: float = 1540.0) -> np.ndarray:
    """计算局部Mach数场。
    
    Mach数: Ma = |v| / c₀
    
    当 Ma > 0.3 时，线性近似失效，需考虑完整的非线性对流方程。
    """
    velocity_magnitude = np.sqrt(flow_u**2 + flow_v**2)
    return velocity_magnitude / c0
