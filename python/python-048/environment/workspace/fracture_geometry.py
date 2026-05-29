"""
fracture_geometry.py
压裂裂缝几何建模模块

原项目映射: 083_bezier_surface (Bezier 曲面片求值)

在水力压裂中，裂缝表面并非理想平面，而是具有曲率的复杂曲面。
采用双三次 Bezier 曲面片对单个裂缝面进行参数化描述，
并在此基础上构建离散裂缝网络（DFN）的拓扑与几何信息。

核心公式:
1. 双三次 Bezier 曲面片:
   S(u,v) = Σ_{i=0}^{3} Σ_{j=0}^{3} B_i^3(u) B_j^3(v) P_{ij}
   其中 Bernstein 基函数:
   B_i^3(u) = C(3,i) u^i (1-u)^{3-i}
   展开形式:
   B_0^3(u) = (1-u)^3
   B_1^3(u) = 3u(1-u)^2
   B_2^3(u) = 3u^2(1-u)
   B_3^3(u) = u^3

2. 曲面法向量（用于确定裂缝面方位）:
   n(u,v) = ∂S/∂u × ∂S/∂v
   ∂S/∂u = Σ_i Σ_j dB_i^3/du B_j^3(v) P_{ij}
   dB_i^3/du = 3 [B_{i-1}^2(u) - B_i^2(u)]  (采用降阶公式)

3. 曲面面积元:
   dA = ||∂S/∂u × ∂S/∂v|| du dv
   裂缝面总面积:
   A = ∫_0^1 ∫_0^1 ||∂S/∂u × ∂S/∂v|| du dv
   该积分通过高阶求积规则（如 prism_jaskowiec_rule 投影到二维）数值计算。

4. 裂缝面到点距离:
   对于空间点 Q，到曲面片 S 的距离定义为:
   d(Q,S) = min_{u,v∈[0,1]} ||Q - S(u,v)||
   通过 Newton-Raphson 迭代或均匀采样网格搜索求解。
"""

import numpy as np
from typing import Tuple, List


def bernstein_basis_3(u: float) -> np.ndarray:
    """
    计算三次 Bernstein 基函数在 u 处的值。

    返回:
        [B0, B1, B2, B3]
    """
    if not (0.0 <= u <= 1.0):
        u = max(0.0, min(1.0, u))
    u2 = u * u
    u3 = u2 * u
    om = 1.0 - u
    om2 = om * om
    om3 = om2 * om
    return np.array([om3,
                     3.0 * u * om2,
                     3.0 * u2 * om,
                     u3])


def bernstein_derivative_3(u: float) -> np.ndarray:
    """
    计算三次 Bernstein 基函数的导数。

    dB_i^3/du = 3 [B_{i-1}^2(u) - B_i^2(u)]
    """
    if not (0.0 <= u <= 1.0):
        u = max(0.0, min(1.0, u))
    return np.array([-3.0 * (1.0 - u) ** 2,
                     3.0 * (1.0 - u) ** 2 - 6.0 * u * (1.0 - u),
                     6.0 * u * (1.0 - u) - 3.0 * u ** 2,
                     3.0 * u ** 2])


def bezier_patch_evaluate(control_points: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    求值双三次 Bezier 曲面片在参数 (u,v) 处的空间坐标。

    参数:
        control_points: (4, 4, 3) 控制点网格 P[i,j,:] = [x,y,z]。
        u, v: 参数坐标，范围 [0,1]。

    返回:
        [x, y, z]
    """
    if control_points.shape != (4, 4, 3):
        raise ValueError("control_points 必须是 (4,4,3) 数组")
    bu = bernstein_basis_3(u)
    bv = bernstein_basis_3(v)
    point = np.zeros(3)
    for i in range(4):
        for j in range(4):
            point += bu[i] * bv[j] * control_points[i, j, :]
    return point


def bezier_patch_normal(control_points: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    计算 Bezier 曲面片在 (u,v) 处的单位法向量。

    公式:
        n = (∂S/∂u × ∂S/∂v) / ||∂S/∂u × ∂S/∂v||
    """
    du = bernstein_derivative_3(u)
    dv = bernstein_derivative_3(v)
    bv = bernstein_basis_3(v)
    bu = bernstein_basis_3(u)

    Su = np.zeros(3)
    Sv = np.zeros(3)
    for i in range(4):
        for j in range(4):
            Su += du[i] * bv[j] * control_points[i, j, :]
            Sv += bu[i] * dv[j] * control_points[i, j, :]

    nvec = np.cross(Su, Sv)
    norm = np.linalg.norm(nvec)
    if norm < 1.0e-14:
        return np.array([0.0, 0.0, 1.0])
    return nvec / norm


def bezier_patch_area(control_points: np.ndarray, n_quad: int = 8) -> float:
    """
    通过均匀网格求积近似计算 Bezier 曲面片的面积。

    公式:
        A ≈ Σ_{k,l} ||∂S/∂u × ∂S/∂v||_{(u_k,v_l)} Δu Δv
    """
    u_vals = np.linspace(0.0, 1.0, n_quad)
    v_vals = np.linspace(0.0, 1.0, n_quad)
    du = 1.0 / (n_quad - 1) if n_quad > 1 else 1.0
    dv = du
    area = 0.0
    for ui in u_vals:
        for vj in v_vals:
            du_basis = bernstein_derivative_3(ui)
            dv_basis = bernstein_derivative_3(vj)
            bv = bernstein_basis_3(vj)
            bu = bernstein_basis_3(ui)
            Su = np.zeros(3)
            Sv = np.zeros(3)
            for i in range(4):
                for j in range(4):
                    Su += du_basis[i] * bv[j] * control_points[i, j, :]
                    Sv += bu[i] * dv_basis[j] * control_points[i, j, :]
            area += np.linalg.norm(np.cross(Su, Sv)) * du * dv
    return float(area)


class FracturePatch:
    """
    单个裂缝曲面片对象，包含几何与控制信息。
    """

    def __init__(self, control_points: np.ndarray, patch_id: int = 0):
        if control_points.shape != (4, 4, 3):
            raise ValueError("控制点维度必须为 (4,4,3)")
        self.cp = np.array(control_points, dtype=float)
        self.id = patch_id
        self._area = None

    def evaluate(self, u: float, v: float) -> np.ndarray:
        return bezier_patch_evaluate(self.cp, u, v)

    def normal(self, u: float, v: float) -> np.ndarray:
        return bezier_patch_normal(self.cp, u, v)

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = bezier_patch_area(self.cp, n_quad=12)
        return self._area

    def centroid(self) -> np.ndarray:
        """近似重心，取四个角点与控制点中心的平均。"""
        return np.mean(self.cp.reshape(-1, 3), axis=0)


def create_planar_fracture_patch(center: np.ndarray, normal: np.ndarray,
                                  length: float, height: float,
                                  patch_id: int = 0) -> FracturePatch:
    """
    基于中心、法向、长宽构建近似平面的 Bezier 曲面片。

    参数:
        center: 裂缝中心 [x, y, z] (m)。
        normal: 法向单位向量 [nx, ny, nz]。
        length: 裂缝长度 (m)。
        height: 裂缝高度 (m)。
        patch_id: 编号。

    返回:
        FracturePatch 对象。
    """
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1.0e-12:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal /= n_norm

    # 构造局部坐标系
    # 找与 normal 不正交的参考向量
    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)

    # 16 个控制点构成平面矩形，轻微扰动以保留曲面特性
    cp = np.zeros((4, 4, 3))
    for i in range(4):
        ui = (i / 3.0) - 0.5
        for j in range(4):
            vj = (j / 3.0) - 0.5
            p = center + ui * length * e1 + vj * height * e2
            # 添加微小弯曲模拟真实裂缝非平面性
            bend = 0.02 * length * np.sin(2.0 * np.pi * ui) * np.cos(2.0 * np.pi * vj)
            p += bend * normal
            cp[i, j, :] = p

    return FracturePatch(cp, patch_id)
