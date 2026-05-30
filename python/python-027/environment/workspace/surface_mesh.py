# -*- coding: utf-8 -*-

import numpy as np


class SurfaceMesh:

    def __init__(self, nodes=None, triangles=None):
        self.nodes = nodes
        self.triangles = triangles
        self.normals = None
        self.areas = None
        if nodes is not None and triangles is not None:
            self._compute_normals_and_areas()

    def generate_flat_plate_mesh(self, width, height, nx, ny, z_offset=0.0):
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


        n_quads = (nx - 1) * (ny - 1)
        n_triangles = 2 * n_quads
        triangles = np.zeros((n_triangles, 3), dtype=int)

        tri_idx = 0
        for j in range(ny - 1):
            for i in range(nx - 1):

                n0 = j * nx + i
                n1 = j * nx + (i + 1)
                n2 = (j + 1) * nx + (i + 1)
                n3 = (j + 1) * nx + i


                triangles[tri_idx] = [n0, n1, n3]
                tri_idx += 1
                triangles[tri_idx] = [n1, n2, n3]
                tri_idx += 1

        self.nodes = nodes
        self.triangles = triangles
        self._compute_normals_and_areas()
        return self

    def generate_cylindrical_mesh(self, radius, height, n_theta, n_z, center=(0.0, 0.0)):
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


            n = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(n)

            if area > 1.0e-20:
                n = n / (2.0 * area)
            else:
                n = np.array([0.0, 0.0, 1.0])

            self.normals[t] = n
            self.areas[t] = area

    def get_triangle_centroids(self):
        if self.nodes is None or self.triangles is None:
            return None
        centroids = np.zeros((len(self.triangles), 3))
        for t in range(len(self.triangles)):
            idx = self.triangles[t]
            centroids[t] = (self.nodes[idx[0]] + self.nodes[idx[1]] + self.nodes[idx[2]]) / 3.0
        return centroids

    def compute_total_area(self):
        if self.areas is None:
            return 0.0
        return np.sum(self.areas)

    def compute_incidence_angles(self, b_field_dir):
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

            cos_theta = max(-1.0, min(1.0, cos_theta))
            angles[t] = np.arccos(abs(cos_theta))

        return angles

    def mesh_quality_stats(self):
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


        min_angles = []
        for t in range(len(self.triangles)):
            idx = self.triangles[t]
            p0, p1, p2 = self.nodes[idx[0]], self.nodes[idx[1]], self.nodes[idx[2]]

            a = np.linalg.norm(p1 - p2)
            b = np.linalg.norm(p0 - p2)
            c = np.linalg.norm(p0 - p1)


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
    mesh = SurfaceMesh()
    mesh.generate_flat_plate_mesh(width=0.05, height=0.05, nx=21, ny=21)
    stats = mesh.mesh_quality_stats()

    print("靶板表面网格统计:")
    for key, val in stats.items():
        print(f"  {key}: {val}")


    b_dir = np.array([0.0, 1.0, 0.1])
    angles = mesh.compute_incidence_angles(b_dir)
    print(f"  平均入射角: {np.degrees(np.mean(angles)):.2f}°")

    return mesh


if __name__ == "__main__":
    demo_mesh()
