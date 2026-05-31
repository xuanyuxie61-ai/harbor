r"""
field_rotation.py
================
磁场拓扑旋转与几何变换模块。
使用四元数描述磁场线在磁重联过程中的三维旋转，
并应用几何变换分析重联区域的对称性。

核心物理与数学模型
------------------
1. 四元数表示三维旋转:
    单位四元数 q = (w, x, y, z) = cos(theta/2) + sin(theta/2) (u_x i + u_y j + u_z k)

    旋转矩阵 R（Rodrigues 公式）:
        R = I + 2s [u]_x + 2s^2 [u]_x^2
    其中 s = sin(theta/2), c = cos(theta/2), [u]_x 为叉积矩阵。

    等价矩阵形式:
        R(1,1) = v1^2 + ca*(1-v1^2)
        R(1,2) = (1-ca)*v1*v2 - sa*v3
        R(1,3) = (1-ca)*v1*v3 + sa*v2
        ...
    其中 ca = cos(theta), sa = sin(theta), v = u/|u|。

2. 磁场线方向的演化:
    在重联过程中，磁场线发生断裂并重新连接，
    其方向变化可用四元数插值描述:
        b(t) = q(t) * b_0 * q(t)^{-1}

3. 对称性变换（对应 pram_view）:
    对于 Harris 电流片，存在反射对称性 y -> -y:
        B_x(-y) = -B_x(y)   (反对称)
        B_y(-y) =  B_y(y)   (对称)

    以及 180 deg 旋转对称性（对应 X-line 中心）:
        R(pi, z) : (x, y) -> (-x, -y)

融入原项目:
- 960_quaternions: 四元数乘法、旋转矩阵转换、轴角表示
- 906_pram_view: 几何变换（旋转、反射、平移）
"""

import numpy as np
from typing import Tuple, Optional


class Quaternion:
    """
    四元数类，封装基本运算。
    """

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.q = np.array([w, x, y, z], dtype=float)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> "Quaternion":
        """
        从旋转轴和角度构造四元数。
        q = [cos(theta/2), sin(theta/2) * axis/|axis|]
        """
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-14:
            raise ValueError("旋转轴不能为零向量")
        u = axis / norm
        w = np.cos(0.5 * angle)
        xyz = np.sin(0.5 * angle) * u
        return cls(w, xyz[0], xyz[1], xyz[2])

    def to_rotation_matrix(self) -> np.ndarray:
        """
        转换为 3x3 旋转矩阵。
        """
        q = self.q / (np.linalg.norm(self.q) + 1e-15)
        w, x, y, z = q
        ca = w ** 2 - x ** 2 - y ** 2 - z ** 2  # cos(theta)
        sa = 2.0 * w * np.sqrt(x ** 2 + y ** 2 + z ** 2)  # sin(theta)
        v_norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if v_norm < 1e-14:
            return np.eye(3)
        v1, v2, v3 = x / v_norm, y / v_norm, z / v_norm

        R = np.zeros((3, 3))
        R[0, 0] = v1 * v1 + ca * (1.0 - v1 * v1)
        R[0, 1] = (1.0 - ca) * v1 * v2 - sa * v3
        R[0, 2] = (1.0 - ca) * v1 * v3 + sa * v2
        R[1, 0] = (1.0 - ca) * v2 * v1 + sa * v3
        R[1, 1] = v2 * v2 + ca * (1.0 - v2 * v2)
        R[1, 2] = (1.0 - ca) * v2 * v3 - sa * v1
        R[2, 0] = (1.0 - ca) * v3 * v1 - sa * v2
        R[2, 1] = (1.0 - ca) * v3 * v2 + sa * v1
        R[2, 2] = v3 * v3 + ca * (1.0 - v3 * v3)
        return R

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """
        四元数乘法（Hamilton 积）。
        q3 = q1 * q2:
            w3 = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x3 = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y3 = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z3 = w1*z2 + x1*y2 - y1*x2 + z1*w2
        """
        a = self.q
        b = other.q
        w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return Quaternion(w, x, y, z)

    def conjugate(self) -> "Quaternion":
        """共轭四元数 q* = (w, -x, -y, -z)。"""
        return Quaternion(self.q[0], -self.q[1], -self.q[2], -self.q[3])

    def norm(self) -> float:
        return np.linalg.norm(self.q)

    def inverse(self) -> "Quaternion":
        n2 = self.norm() ** 2
        if n2 < 1e-28:
            raise ZeroDivisionError("四元数模为零")
        c = self.conjugate()
        return Quaternion(c.q[0] / n2, c.q[1] / n2, c.q[2] / n2, c.q[3] / n2)

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        用四元数旋转向量: v' = q * v * q^{-1}。
        """
        v = np.asarray(v, dtype=float)
        vq = Quaternion(0.0, v[0], v[1], v[2])
        q_inv = self.inverse()
        result = self * vq * q_inv
        return result.q[1:]

    def slerp(self, other: "Quaternion", t: float) -> "Quaternion":
        """
        球面线性插值（Spherical Linear Interpolation）。
        用于磁场线方向的平滑过渡。
        """
        if t < 0.0 or t > 1.0:
            raise ValueError("t 必须在 [0, 1] 之间")
        q1 = self.q / (self.norm() + 1e-15)
        q2 = other.q / (other.norm() + 1e-15)
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        if dot > 0.9995:
            # 近似线性插值
            qm = q1 + t * (q2 - q1)
            qm = qm / (np.linalg.norm(qm) + 1e-15)
            return Quaternion(*qm)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        q3 = q2 - q1 * dot
        q3 = q3 / (np.linalg.norm(q3) + 1e-15)
        q_out = q1 * np.cos(theta) + q3 * np.sin(theta)
        return Quaternion(*q_out)


class MagneticTopology:
    """
    磁场拓扑变换分析。
    """

    @staticmethod
    def reflect_y(v: np.ndarray) -> np.ndarray:
        """
        关于 y=0 平面的反射变换:
            (x, y, z) -> (-x, y, -z)
        对应 Harris 电流片的反对称性。
        """
        v = np.asarray(v, dtype=float)
        R = np.diag([-1.0, 1.0, -1.0])
        return R @ v

    @staticmethod
    def rotate_z_180(v: np.ndarray) -> np.ndarray:
        """
        绕 z 轴旋转 180 deg:
            (x, y, z) -> (-x, -y, z)
        对应 X-line 中心的对称性。
        """
        v = np.asarray(v, dtype=float)
        q = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)
        return q.rotate_vector(v)

    @staticmethod
    def translate_x(v: np.ndarray, dx: float) -> np.ndarray:
        """平移变换。"""
        v = np.asarray(v, dtype=float)
        vt = v.copy()
        vt[0] += dx
        return vt

    @staticmethod
    def check_harris_symmetry(B_field_func: callable,
                               y_points: np.ndarray) -> dict:
        """
        验证 Harris 电流片的反射对称性。
        要求: B_x(-y) = -B_x(y), B_y(-y) = B_y(y)
        """
        y = np.asarray(y_points, dtype=float)
        Bp = B_field_func(y)
        Bn = B_field_func(-y)
        err_Bx = np.max(np.abs(Bp[:, 0] + Bn[:, 0]))
        err_By = np.max(np.abs(Bp[:, 1] - Bn[:, 1]))
        return {
            'max_Bx_antisym_error': err_Bx,
            'max_By_sym_error': err_By,
            'symmetric': err_Bx < 1e-10 and err_By < 1e-10
        }


def demo_field_rotation():
    """
    演示四元数旋转与磁场拓扑变换。
    """
    print("\n[FieldRotation] 演示: 四元数磁场旋转")

    # 初始磁场方向（沿 x 轴）
    B0 = np.array([1.0, 0.0, 0.0])
    # 旋转 90 deg 绕 z 轴（模拟重联中磁场线转向）
    q90 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    B1 = q90.rotate_vector(B0)
    print(f"  B0 = {B0}")
    print(f"  旋转 90 deg 后 B1 = {B1}")

    # SLERP 插值
    q0 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.0)
    q1 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)
    print(f"  SLERP 插值验证 (t=0.5, 应旋转 90 deg):")
    q_mid = q0.slerp(q1, 0.5)
    B_mid = q_mid.rotate_vector(B0)
    print(f"    B_mid = {B_mid}")

    # 旋转矩阵验证
    R = q90.to_rotation_matrix()
    B1_mat = R @ B0
    print(f"  旋转矩阵结果: {B1_mat}")

    print("\n[FieldRotation] 演示: Harris 对称性变换")
    from harris_equilibrium import HarrisEquilibrium
    eq = HarrisEquilibrium()
    y_test = np.linspace(-eq.y_max, eq.y_max, 21)
    sym = MagneticTopology.check_harris_symmetry(eq.B_field, y_test)
    print(f"  B_x 反对称误差: {sym['max_Bx_antisym_error']:.3e}")
    print(f"  B_y 对称误差: {sym['max_By_sym_error']:.3e}")
    print(f"  是否满足 Harris 对称性: {sym['symmetric']}")


if __name__ == "__main__":
    demo_field_rotation()
