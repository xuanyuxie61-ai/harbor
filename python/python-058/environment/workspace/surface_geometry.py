"""
地表几何与通量积分模块 (Surface Geometry & Flux Integration)

集成种子项目:
- 1315_triangle_svg: 三角形几何计算 (去除可视化, 保留几何数学)

科学背景:
  中尺度对流系统的地表通量 (感热、潜热、动量) 通常在地表三角形网格上积分.
  使用三角形面积公式和重心坐标进行精确的通量面积分:

    Φ_total = ∫_S φ(x,y) dS ≈ Σ_triangles A_i * φ(x_c, y_c)

核心公式:
  三角形面积 (Heron 公式 / 叉积):
    A = 0.5 * |(v1-v0) × (v2-v0)|
       = 0.5 * |x1(y2-y0) + x2(y0-y1) + x0(y1-y2)|

  重心坐标插值:
    P = λ0 v0 + λ1 v1 + λ2 v2,  λ0+λ1+λ2=1
"""

import numpy as np
from typing import List, Tuple


def triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算三维空间中三角形的面积 (基于 1315_triangle_svg 的几何核心).

    A = 0.5 * || (v1-v0) × (v2-v0) ||
    """
    v0, v1, v2 = np.asarray(v0), np.asarray(v1), np.asarray(v2)
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross)


def triangle_centroid(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """三角形重心."""
    return (np.asarray(v0) + np.asarray(v1) + np.asarray(v2)) / 3.0


def barycentric_coordinates(p: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                            v2: np.ndarray) -> Tuple[float, float, float]:
    """
    计算点 p 关于三角形 (v0,v1,v2) 的重心坐标.

    λ0 = A(p,v1,v2) / A(v0,v1,v2)
    λ1 = A(v0,p,v2) / A(v0,v1,v2)
    λ2 = A(v0,v1,p) / A(v0,v1,v2)
    """
    v0, v1, v2, p = np.asarray(v0), np.asarray(v1), np.asarray(v2), np.asarray(p)
    A_total = triangle_area(v0, v1, v2)
    if A_total < 1e-20:
        return 1.0, 0.0, 0.0
    A0 = triangle_area(p, v1, v2)
    A1 = triangle_area(v0, p, v2)
    A2 = triangle_area(v0, v1, p)
    return A0 / A_total, A1 / A_total, A2 / A_total


def point_in_triangle(p: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                      v2: np.ndarray, tol: float = 1e-10) -> bool:
    """判断点是否在三角形内 (含边界)."""
    l0, l1, l2 = barycentric_coordinates(p, v0, v1, v2)
    return (l0 >= -tol) and (l1 >= -tol) and (l2 >= -tol) and abs(l0 + l1 + l2 - 1.0) < tol


def integrate_over_triangles(vertices_list: List[np.ndarray],
                              flux_func) -> float:
    """
    在三角形列表上积分标量通量.

    参数:
      vertices_list: 每个元素为 (3,2) 或 (3,3) 的三角形顶点
      flux_func: 以重心坐标或物理坐标为输入的通量函数

    返回:
      总通量积分值.
    """
    total = 0.0
    for verts in vertices_list:
        v0, v1, v2 = verts[0], verts[1], verts[2]
        A = triangle_area(v0, v1, v2)
        if A < 1e-20:
            continue
        centroid = triangle_centroid(v0, v1, v2)
        try:
            flux_val = flux_func(centroid)
            if np.isfinite(flux_val):
                total += A * flux_val
        except Exception:
            continue
    return total


def regular_surface_mesh(nx: int, ny: int, xlim: Tuple[float, float],
                         ylim: Tuple[float, float]) -> List[np.ndarray]:
    """
    生成规则矩形区域上的三角形表面网格 (每个矩形划分为 2 个三角形).

    返回三角形顶点列表.
    """
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p00 = np.array([x[i], y[j], 0.0])
            p10 = np.array([x[i+1], y[j], 0.0])
            p01 = np.array([x[i], y[j+1], 0.0])
            p11 = np.array([x[i+1], y[j+1], 0.0])
            triangles.append(np.array([p00, p10, p11]))
            triangles.append(np.array([p00, p11, p01]))
    return triangles


def surface_sensible_heat_flux(t_sfc: float, t_air: float,
                                wind_speed: float,
                                drag_coeff: float = 1.2e-3,
                                rho: float = 1.225,
                                cp: float = 1004.0) -> float:
    """
    地表感热通量 (W/m²):
      H = ρ * cp * C_d * |V| * (T_sfc - T_air)
    """
    return rho * cp * drag_coeff * wind_speed * (t_sfc - t_air)


def surface_latent_heat_flux(q_sfc: float, q_air: float,
                              wind_speed: float,
                              drag_coeff: float = 1.2e-3,
                              rho: float = 1.225,
                              Lv: float = 2.501e6) -> float:
    """
    地表潜热通量 (W/m²):
      LE = ρ * Lv * C_d * |V| * (q_sfc - q_air)
    """
    return rho * Lv * drag_coeff * wind_speed * (q_sfc - q_air)
