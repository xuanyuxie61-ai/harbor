"""
距离函数网格生成器
基于 distmesh 核心算法：有符号距离函数 (SDF)、力平衡迭代、Delaunay 三角化、边界投影。

在蛋白质折叠中的应用：
- 在反应坐标空间 (如 Q-RMSD 平面) 生成自适应质量网格
- 分子表面/口袋的三角化离散
- 粗粒化模型初始粒子排布
- Poisson-Boltzmann 方程的有限元网格

数学基础:
    力平衡方程:
        每条边视为弹簧，施加排斥/吸引力使边长趋近目标长度 L0
        F = max(L0 - L, 0)
        F_vec = (F / L) * (p_j - p_i)
    
    目标长度:
        L0 = h_bar * F_scale * sqrt( Σ L² / Σ h_bar² )
    
    边界投影:
        p_new = p - d(p) * ∇d(p) / |∇d(p)|
    
    其中 d(p) 为有符号距离函数，负值表示内部。
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Callable, Tuple, Optional


def signed_distance_circle(p: np.ndarray, xc: float, yc: float, r: float) -> np.ndarray:
    """
    圆的有符号距离函数。
    
    d(p) = |p - c| - r
    d < 0: 内部；d > 0: 外部；d = 0: 边界。
    
    Parameters
    ----------
    p : np.ndarray, shape (N, 2)
        点坐标。
    xc, yc : float
        圆心。
    r : float
        半径。
    
    Returns
    -------
    d : np.ndarray, shape (N,)
        有符号距离。
    """
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def signed_distance_rectangle(p: np.ndarray, xlim: Tuple[float, float],
                              ylim: Tuple[float, float]) -> np.ndarray:
    """
    矩形的有符号距离函数。
    
    Parameters
    ----------
    p : np.ndarray, shape (N, 2)
        点坐标。
    xlim : tuple
        x 方向边界 (xmin, xmax)。
    ylim : tuple
        y 方向边界 (ymin, ymax)。
    
    Returns
    -------
    d : np.ndarray
        有符号距离。
    """
    dx = np.maximum(xlim[0] - p[:, 0], p[:, 0] - xlim[1])
    dy = np.maximum(ylim[0] - p[:, 1], p[:, 1] - ylim[1])
    d_inside = np.maximum(dx, dy)
    # 对于外部点
    d_outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
    d = np.where(np.logical_and(dx < 0, dy < 0), d_inside, d_outside)
    return d


def distmesh_2d(fd: Callable[[np.ndarray], np.ndarray],
                fh: Callable[[np.ndarray], np.ndarray],
                h0: float,
                bbox: Tuple[Tuple[float, float], Tuple[float, float]],
                pfix: Optional[np.ndarray] = None,
                iteration_max: int = 100,
                tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D 距离函数网格生成器 (Persson & Strang 算法)。
    
    算法步骤:
        1. 在边界框内均匀撒点
        2. 根据密度函数 fh 进行接受-拒绝采样
        3. 迭代:
           a. Delaunay 三角化
           b. 计算每条边的目标长度和弹簧力
           c. 更新节点位置
           d. 将越界点投影回边界
           e. 检查收敛
    
    Parameters
    ----------
    fd : callable
        有符号距离函数，输入 (N,2) 输出 (N,)。
    fh : callable
        网格尺寸密度函数，输入 (N,2) 输出 (N,)。
    h0 : float
        目标边长。
    bbox : tuple
        边界框 ((xmin, xmax), (ymin, ymax))。
    pfix : np.ndarray, optional
        固定点坐标。
    iteration_max : int
        最大迭代次数。
    tol : float
        收敛阈值（最大位移 < tol * h0）。
    
    Returns
    -------
    p : np.ndarray, shape (N, 2)
        网格节点。
    t : np.ndarray, shape (Nt, 3)
        三角形单元（0-based）。
    """
    (xmin, xmax), (ymin, ymax) = bbox
    
    # 初始均匀撒点
    x_range = np.arange(xmin, xmax, h0)
    y_range = np.arange(ymin, ymax, h0 * np.sqrt(3) / 2)
    xx, yy = np.meshgrid(x_range, y_range)
    # 交错网格
    yy[1::2, :] += h0 / 2
    p_init = np.column_stack((xx.ravel(), yy.ravel()))
    
    # 接受-拒绝采样 (根据密度函数)
    if fh is not None:
        h_vals = fh(p_init)
        h_vals = np.maximum(h_vals, 1e-6)
        prob = np.minimum(1.0, (h0 / h_vals) ** 2)
        rng = np.random.default_rng(42)
        accept = rng.random(len(p_init)) < prob
        p = p_init[accept]
    else:
        p = p_init.copy()
    
    # 保留固定点
    if pfix is not None and len(pfix) > 0:
        p = np.vstack([pfix, p])
    
    # 去除外部点
    d = fd(p)
    p = p[d < 0]
    
    F_scale = 1.2  # 力平衡缩放因子
    
    for it in range(iteration_max):
        if len(p) < 3:
            raise ValueError("Too few points for triangulation")
        
        tri = Delaunay(p)
        t = tri.simplices
        
        # 提取边（每个三角形3条边）
        edges = []
        for tri_elem in t:
            edges.extend([
                tuple(sorted([tri_elem[0], tri_elem[1]])),
                tuple(sorted([tri_elem[1], tri_elem[2]])),
                tuple(sorted([tri_elem[2], tri_elem[0]])),
            ])
        edges = list(set(edges))
        
        # 计算边长和力
        forces = np.zeros_like(p)
        bar_lengths = []
        for i, j in edges:
            dp = p[j] - p[i]
            L = np.linalg.norm(dp)
            if L < 1e-12:
                continue
            bar_lengths.append(L)
            
            # 目标长度
            hi = fh(np.array([p[i]]))[0] if fh is not None else h0
            hj = fh(np.array([p[j]]))[0] if fh is not None else h0
            h_bar = 0.5 * (hi + hj)
            L0 = h_bar * F_scale
            
            # 弹簧力
            F = max(L0 - L, 0.0)
            fvec = (F / L) * dp
            forces[i] -= fvec
            forces[j] += fvec
        
        # 更新位置 (阻尼)
        p_new = p + 0.2 * forces
        
        # 边界投影
        d_new = fd(p_new)
        # 数值梯度 (有限差分)
        eps = 1e-6
        grad = np.zeros_like(p_new)
        for dim in range(2):
            p_perturb = p_new.copy()
            p_perturb[:, dim] += eps
            grad[:, dim] = (fd(p_perturb) - d_new) / eps
        
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.maximum(grad_norm, 1e-12)
        
        # 只投影外部点
        outside = d_new > 0
        p_new[outside] -= d_new[outside][:, None] * (grad[outside] / grad_norm[outside])
        
        # 固定点约束
        if pfix is not None and len(pfix) > 0:
            p_new[:len(pfix)] = pfix
        
        # 收敛检查
        max_disp = np.max(np.linalg.norm(p_new - p, axis=1))
        p = p_new
        if max_disp < tol * h0:
            break
    
    # 最终三角化
    tri = Delaunay(p)
    t = tri.simplices
    return p, t


def simpqual(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    计算三角形网格质量（内切圆半径 / 外接圆半径）。
    
    质量范围 [0, 1]，1 为等边三角形（最优）。
    
    Parameters
    ----------
    p : np.ndarray, shape (N, 2)
        节点坐标。
    t : np.ndarray, shape (Nt, 3)
        三角形单元。
    
    Returns
    -------
    quality : np.ndarray, shape (Nt,)
        每个三角形的质量指标。
    """
    quality = np.zeros(len(t))
    for i, tri in enumerate(t):
        v1, v2, v3 = p[tri[0]], p[tri[1]], p[tri[2]]
        a = np.linalg.norm(v2 - v1)
        b = np.linalg.norm(v3 - v2)
        c = np.linalg.norm(v1 - v3)
        
        # 面积 (Heron 公式)
        s = 0.5 * (a + b + c)
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        
        # 外接圆半径 R = abc / (4*area)
        R = (a * b * c) / (4.0 * max(area, 1e-12))
        # 内切圆半径 r = area / s
        r = area / max(s, 1e-12)
        
        quality[i] = r / max(R, 1e-12)
    return quality


def generate_reaction_coordinate_mesh(q_range: Tuple[float, float],
                                      rmsd_range: Tuple[float, float],
                                      h0: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    在反应坐标空间 (Q, RMSD) 上生成自适应网格。
    
    策略:
        使用矩形边界，在中间区域 (Q≈0.5) 加密网格以捕捉过渡态。
    
    Parameters
    ----------
    q_range : tuple
        Q 坐标范围 (min, max)。
    rmsd_range : tuple
        RMSD 坐标范围 (min, max)。
    h0 : float
        目标边长。
    
    Returns
    -------
    p : np.ndarray
        网格节点 (Q, RMSD)。
    t : np.ndarray
        三角形单元。
    """
    (qmin, qmax), (rmsdmin, rmsdmax) = q_range, rmsd_range
    bbox = ((qmin, qmax), (rmsdmin, rmsdmax))
    
    # 密度函数：在 Q=0.5 附近加密
    def fh(pp):
        q = pp[:, 0]
        rmsd = pp[:, 1]
        center_q = 0.5 * (qmin + qmax)
        dist = np.sqrt((q - center_q) ** 2 + (rmsd - 0.5 * (rmsdmin + rmsdmax)) ** 2)
        return h0 * (0.5 + 0.5 * dist / max(qmax - qmin, rmsdmax - rmsdmin))
    
    def fd(pp):
        return signed_distance_rectangle(pp, (qmin, qmax), (rmsdmin, rmsdmax))
    
    p, t = distmesh_2d(fd, fh, h0, bbox, iteration_max=80, tol=1e-3)
    return p, t
