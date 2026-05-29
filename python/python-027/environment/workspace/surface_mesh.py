# -*- coding: utf-8 -*-
"""
surface_mesh.py
壁材料表面网格生成与处理模块
基于种子项目 823_obj_to_tri_surface (三角剖分) 和 1201_tensor_grid_display (张量积网格) 重构

本模块生成托卡马克偏滤器靶板表面的三角网格，用于离子轰击和侵蚀计算。
靶板表面通常具有复杂的3D几何形状，需要高质量的三角剖分。
"""

import numpy as np


class SurfaceMesh:
    """
    壁材料表面三角网格类
    
    生成并管理靶板表面的三角网格，支持:
        - 结构化四边形到三角形的转换
        - 张量积网格生成
        - 表面法向计算
        - 网格质量评估
    """

    def __init__(self, nodes=None, triangles=None):
        """
        Parameters:
            nodes:     (n_nodes, 3) 节点坐标 [m]
            triangles: (n_triangles, 3) 三角形顶点索引
        """
        self.nodes = nodes
        self.triangles = triangles
        self.normals = None
        self.areas = None
        if nodes is not None and triangles is not None:
            self._compute_normals_and_areas()

    def generate_flat_plate_mesh(self, width, height, nx, ny, z_offset=0.0):
        """
        生成平板靶板表面的结构化三角网格
        
        基于 tensor_grid_display 的张量积网格思想:
            - 先生成 nx x ny 的笛卡尔网格
            - 再将每个四边形分裂为2个三角形
        
        Parameters:
            width:     板宽度 [m]
            height:    板高度 [m]
            nx, ny:    x, y 方向节点数
            z_offset:  z方向偏移 [m]
        
        Returns:
            self
        """
        if nx < 2 or ny < 2:
            raise ValueError("nx 和 ny 必须至少为 2")
        if width <= 0 or height <= 0:
            raise ValueError("width 和 height 必须为正")

        x = np.linspace(-width/2, width/2, nx)
        y = np.linspace(-height/2, height/2, ny)

        n_nodes = nx * ny
        nodes = np.zeros((n_nodes, 3))

        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                nodes[idx, 0] = x[i]
                nodes[idx, 1] = y[j]
                nodes[idx, 2] = z_offset

        # 四边形到三角形转换（基于 faces_to_triangles 思想）
        n_quads = (nx - 1) * (ny - 1)
        n_triangles = 2 * n_quads
        triangles = np.zeros((n_triangles, 3), dtype=int)

        tri_idx = 0
        for j in range(ny - 1):
            for i in range(nx - 1):
                # 四边形四个顶点
                n0 = j * nx + i
                n1 = j * nx + (i + 1)
                n2 = (j + 1) * nx + (i + 1)
                n3 = (j + 1) * nx + i

                # 分裂为两个三角形
                triangles[tri_idx] = [n0, n1, n3]
                tri_idx += 1
                triangles[tri_idx] = [n1, n2, n3]
                tri_idx += 1

        self.nodes = nodes
        self.triangles = triangles
        self._compute_normals_and_areas()
        return self

    def generate_cylindrical_mesh(self, radius, height, n_theta, n_z, center=(0.0, 0.0)):
        """
        生成圆柱面靶板网格（模拟偏滤器 dome 结构）
        
        使用张量积参数化:
            x = r * cos(theta)
            y = r * sin(theta)
            z = z
        """
        if n_theta < 3 or n_z < 2:
            raise ValueError("n_theta >= 3, n_z >= 2")
        if radius <= 0 or height <= 0:
            raise ValueError("radius 和 height 必须为正")

        theta = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=False)
        z = np.linspace(-height/2, height/2, n_z)

        n_nodes = n_theta * n_z
        nodes = np.zeros((n_nodes, 3))

        for j in range(n_z):
            for i in range(n_theta):
                idx = j * n_theta + i
                nodes[idx, 0] = center[0] + radius * np.cos(theta[i])
                nodes[idx, 1] = center[1] + radius * np.sin(theta[i])
                nodes[idx, 2] = z[j]

        n_triangles = 2 * n_theta * (n_z - 1)
        triangles = np.zeros((n_triangles, 3), dtype=int)

        tri_idx = 0
        for j in range(n_z - 1):
            for i in range(n_theta):
                n0 = j * n_theta + i
                n1 = j * n_theta + ((i + 1) % n_theta)
                n2 = (j + 1) * n_theta + ((i + 1) % n_theta)
                n3 = (j + 1) * n_theta + i

                triangles[tri_idx] = [n0, n1, n3]
                tri_idx += 1
                triangles[tri_idx] = [n1, n2, n3]
                tri_idx += 1

        self.nodes = nodes
        self.triangles = triangles
        self._compute_normals_and_areas()
        return self

    def _compute_normals_and_areas(self):
        """计算每个三角形的法向量和面积"""
        if self.nodes is None or self.triangles is None:
            return

        n_tri = len(self.triangles)
        self.normals = np.zeros((n_tri, 3))
        self.areas = np.zeros(n_tri)

        for t in range(n_tri):
            idx = self.triangles[t]
            p0 = self.nodes[idx[0]]
            p1 = self.nodes[idx[1]]
            p2 = self.nodes[idx[2]]

            v1 = p1 - p0
            v2 = p2 - p0

            # 叉积得到法向
            n = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(n)

            if area > 1.0e-20:
                n = n / (2.0 * area)
            else:
                n = np.array([0.0, 0.0, 1.0])

            self.normals[t] = n
            self.areas[t] = area

    def get_triangle_centroids(self):
        """计算三角形重心"""
        if self.nodes is None or self.triangles is None:
            return None
        centroids = np.zeros((len(self.triangles), 3))
        for t in range(len(self.triangles)):
            idx = self.triangles[t]
            centroids[t] = (self.nodes[idx[0]] + self.nodes[idx[1]] + self.nodes[idx[2]]) / 3.0
        return centroids

    def compute_total_area(self):
        """计算总表面积"""
        if self.areas is None:
            return 0.0
        return np.sum(self.areas)

    def compute_incidence_angles(self, b_field_dir):
        """
        计算每个三角形的离子入射角
        
        入射角 theta 满足: cos(theta) = -B_hat · n_hat
        （假设离子沿磁场线运动到达靶板）
        
        Parameters:
            b_field_dir: 磁场方向单位向量 (3,)
        
        Returns:
            angles: 入射角数组 [rad]
        """
        if self.normals is None:
            return None

        b = np.asarray(b_field_dir, dtype=float)
        b_norm = np.linalg.norm(b)
        if b_norm < 1.0e-20:
            return np.zeros(len(self.normals))
        b = b / b_norm

        angles = np.zeros(len(self.normals))
        for t in range(len(self.normals)):
            cos_theta = -np.dot(b, self.normals[t])
            # 限制在 [-1, 1]
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angles[t] = np.arccos(abs(cos_theta))

        return angles

    def mesh_quality_stats(self):
        """
        评估网格质量
        
        返回:
            - 最小/最大/平均面积
            - 最小角度
            - 长宽比统计
        """
        if self.areas is None or self.triangles is None:
            return {}

        stats = {
            'n_nodes': len(self.nodes),
            'n_triangles': len(self.triangles),
            'total_area': self.compute_total_area(),
            'min_area': np.min(self.areas),
            'max_area': np.max(self.areas),
            'mean_area': np.mean(self.areas),
        }

        # 计算三角形最小角
        min_angles = []
        for t in range(len(self.triangles)):
            idx = self.triangles[t]
            p0, p1, p2 = self.nodes[idx[0]], self.nodes[idx[1]], self.nodes[idx[2]]

            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p0 - p2)
            c = np.linalg.norm(p0 - p1)

            # 余弦定理求角
            if a > 0 and b > 0 and c > 0:
                cos_A = min(1.0, max(-1.0, (b*b + c*c - a*a) / (2*b*c)))
                cos_B = min(1.0, max(-1.0, (a*a + c*c - b*b) / (2*a*c)))
                cos_C = min(1.0, max(-1.0, (a*a + b*b - c*c) / (2*a*b)))
                angles = [np.arccos(cos_A), np.arccos(cos_B), np.arccos(cos_C)]
                min_angles.append(min(angles))

        if min_angles:
            stats['min_triangle_angle_deg'] = np.degrees(min(min_angles))
            stats['mean_triangle_angle_deg'] = np.degrees(np.mean(min_angles))

        return stats


def demo_mesh():
    """演示网格生成"""
    mesh = SurfaceMesh()
    mesh.generate_flat_plate_mesh(width=0.05, height=0.05, nx=21, ny=21)
    stats = mesh.mesh_quality_stats()

    print("靶板表面网格统计:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    # 计算入射角
    b_dir = np.array([0.0, 1.0, 0.1])
    angles = mesh.compute_incidence_angles(b_dir)
    print(f"  平均入射角: {np.degrees(np.mean(angles)):.2f}°")

    return mesh


if __name__ == "__main__":
    demo_mesh()
