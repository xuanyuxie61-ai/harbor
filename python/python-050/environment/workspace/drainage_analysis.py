"""
drainage_analysis.py
冰盖排水流域连通分量分析

基于种子项目 205_components 的连通分量标记算法，
应用于冰盖表面/底部高程场以识别独立的排水流域 (catchments) 与冰流分支。

核心数学:
  1. 流向计算 (D8 算法简化):
       对每个网格点，比较 8 邻域内的最大下坡梯度:
       \nabla_{ij} = \frac{s_{neighbor} - s_{ij}}{d_{ij}}

       其中 d_{ij} 为网格间距 (正对角线: dx/dy, 对角线: \sqrt{dx^2+dy^2})。

  2. 汇水区 (Catchment) 定义:
       汇水区为所有流向同一出口点的网格集合。

  3. 连通分量标记 (Connected Component Labeling):
       对二值化后的流域掩码，使用栈式 DFS 标记 4-连通或 8-连通区域。

  4. 水文连通性指标:
       - 流域面积 A_c = N_{cell} \cdot dx \cdot dy
       - 形心位置: \bar{x} = \frac{1}{A_c} \sum x_i A_i
       - 高程积分 (Hypsometric Integral):
           HI = \frac{\bar{h} - h_{min}}{h_{max} - h_{min}}

应用场景:
  - 识别南极冰盖的主要冰流分支 (如 Byrd Glacier, Lambert Glacier)
  - 分析冰盖表面融化水的径流路径
  - 评估冰架崩解对排水格局的影响
"""

import numpy as np
from typing import List, Tuple, Dict


def compute_flow_direction(surface: np.ndarray,
                           dx: float,
                           dy: float) -> np.ndarray:
    """
    计算每个网格点的流向 (D8 简化版，8 方向编码)。

    编码:
        0: 东, 1: 东南, 2: 南, 3: 西南,
        4: 西, 5: 西北, 6: 北, 7: 东北,
        -1: 局部极小值 (汇点)

    参数:
        surface: 表面高程场 (ny, nx)
        dx, dy: 网格间距 (m)

    返回:
        flow_dir: (ny, nx) 流向编码
    """
    surface = np.asarray(surface, dtype=np.float64)
    ny, nx = surface.shape

    flow_dir = np.full((ny, nx), -1, dtype=np.int32)

    # 8 邻域偏移与距离
    offsets = [
        (0, 1, dx),      # 东
        (1, 1, np.sqrt(dx**2 + dy**2)),  # 东南
        (1, 0, dy),      # 南
        (1, -1, np.sqrt(dx**2 + dy**2)), # 西南
        (0, -1, dx),     # 西
        (-1, -1, np.sqrt(dx**2 + dy**2)),# 西北
        (-1, 0, dy),     # 北
        (-1, 1, np.sqrt(dx**2 + dy**2)), # 东北
    ]

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            h0 = surface[i, j]
            max_slope = -1e20
            best_dir = -1

            for code, (di, dj, dist) in enumerate(offsets):
                h_neighbor = surface[i + di, j + dj]
                slope = (h0 - h_neighbor) / dist
                if slope > max_slope:
                    max_slope = slope
                    best_dir = code

            # 若周围都比自身高，则为汇点
            if max_slope <= 0:
                best_dir = -1

            flow_dir[i, j] = best_dir

    return flow_dir


def label_connected_components_2d(mask: np.ndarray,
                                   connectivity: int = 4) -> Tuple[np.ndarray, int]:
    """
    对二值掩码进行二维连通分量标记 (栈式 DFS)。

    基于种子项目 205_components 的 flood-fill 思想。

    参数:
        mask: 二值数组 (ny, nx)，非零值视为前景
        connectivity: 4 或 8 连通

    返回:
        labels: (ny, nx) 标记数组 (0 为背景)
        n_components: 连通分量数
    """
    mask = np.asarray(mask, dtype=np.bool_)
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    label_id = 0

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(ny):
        for j in range(nx):
            if mask[i, j] and labels[i, j] == 0:
                label_id += 1
                stack = [(i, j)]
                labels[i, j] = label_id

                while stack:
                    ci, cj = stack.pop()
                    for di, dj in neighbors:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < ny and 0 <= nj < nx:
                            if mask[ni, nj] and labels[ni, nj] == 0:
                                labels[ni, nj] = label_id
                                stack.append((ni, nj))

    return labels, label_id


def identify_catchments(surface: np.ndarray,
                         mask: np.ndarray,
                         dx: float,
                         dy: float,
                         min_area: float = 1e6) -> Dict[int, Dict]:
    """
    识别冰盖表面的独立排水流域。

    参数:
        surface: 表面高程场 (ny, nx)
        mask: 冰覆盖掩码 (ny, nx)
        dx, dy: 网格间距
        min_area: 最小流域面积阈值 (m^2)

    返回:
        catchments: 字典 {label_id: {area, centroid, mean_elev, ...}}
    """
    labels, n_comp = label_connected_components_2d(mask, connectivity=4)
    catchments = {}

    for comp_id in range(1, n_comp + 1):
        comp_mask = (labels == comp_id)
        n_cells = int(np.sum(comp_mask))
        area = n_cells * dx * dy

        if area < min_area:
            continue

        indices = np.argwhere(comp_mask)
        ys = indices[:, 0]
        xs = indices[:, 1]

        centroid_x = float(np.mean(xs)) * dx
        centroid_y = float(np.mean(ys)) * dy
        mean_elev = float(np.mean(surface[comp_mask]))
        min_elev = float(np.min(surface[comp_mask]))
        max_elev = float(np.max(surface[comp_mask]))

        # 高程积分
        if max_elev > min_elev:
            hypsometric_integral = (mean_elev - min_elev) / (max_elev - min_elev)
        else:
            hypsometric_integral = 0.5

        catchments[comp_id] = {
            'area_m2': area,
            'n_cells': n_cells,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'mean_elevation': mean_elev,
            'min_elevation': min_elev,
            'max_elevation': max_elev,
            'hypsometric_integral': hypsometric_integral,
        }

    return catchments


def compute_drainage_density(catchments: Dict[int, Dict],
                              total_ice_area: float) -> float:
    """
    计算排水密度:

        D_d = N_{catchments} / A_{total}
    """
    if total_ice_area <= 0:
        return 0.0
    return len(catchments) / total_ice_area


def merge_small_catchments(labels: np.ndarray,
                           catchments: Dict[int, Dict],
                           min_area: float) -> np.ndarray:
    """
    将小于面积阈值的流域合并到最近的邻域。
    """
    new_labels = labels.copy()
    ny, nx = labels.shape

    for comp_id, info in catchments.items():
        if info['area_m2'] < min_area:
            # 将该分量设为 0 (背景)
            new_labels[labels == comp_id] = 0

    return new_labels


def extract_main_flow_branches(surface: np.ndarray,
                                thickness: np.ndarray,
                                dx: float, dy: float,
                                velocity_threshold: float = 10.0) -> Dict[int, Dict]:
    """
    从厚度与表面高程提取主要冰流分支 (基于厚度与坡度的综合指标)。

    冰流强度指标:
        I_{flow} = H^{n+1} |\nabla s|^{n-1}

    参数:
        surface: 表面高程
        thickness: 冰厚度
        dx, dy: 间距
        velocity_threshold: 强度阈值

    返回:
        branches: 分支信息字典
    """
    ny, nx = surface.shape
    n = 3.0

    # 计算梯度模
    grad_x = np.zeros_like(surface)
    grad_y = np.zeros_like(surface)
    grad_x[:, 1:-1] = (surface[:, 2:] - surface[:, :-2]) / (2.0 * dx)
    grad_y[1:-1, :] = (surface[2:, :] - surface[:-2, :]) / (2.0 * dy)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.maximum(grad_mag, 1e-12)

    intensity = (thickness ** (n + 1.0)) * (grad_mag ** (n - 1.0))
    mask = intensity > velocity_threshold

    branches = identify_catchments(surface, mask, dx, dy, min_area=dx*dy*10)
    return branches
