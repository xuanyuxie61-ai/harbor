# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional


class OBJGeometryParser:

    def __init__(self):
        self.vertices = np.zeros((0, 3))
        self.faces = []
        self.normals = np.zeros((0, 3))
        self.face_normals = []
        self.face_areas = []

    def parse_string(self, obj_text: str) -> dict:
        vertices = []
        normals = []
        faces = []

        lines = obj_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('$'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            keyword = parts[0].upper()

            if keyword == 'V' and len(parts) >= 4:

                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                except (ValueError, IndexError):
                    continue

            elif keyword == 'VN' and len(parts) >= 4:

                try:
                    nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                    normals.append([nx, ny, nz])
                except (ValueError, IndexError):
                    continue

            elif keyword == 'F':

                face_indices = []
                for part in parts[1:]:

                    idx_str = part.split('/')[0]
                    try:
                        idx = int(idx_str)

                        face_indices.append(idx - 1 if idx > 0 else idx)
                    except ValueError:
                        continue
                if len(face_indices) >= 3:
                    faces.append(face_indices)

        self.vertices = np.array(vertices)
        self.normals = np.array(normals)
        self.faces = faces


        self._compute_face_properties()

        return {
            "n_vertices": len(vertices),
            "n_faces": len(faces),
            "n_normals": len(normals),
            "vertices": self.vertices,
            "faces": self.faces,
            "face_areas": np.array(self.face_areas),
            "face_normals": np.array(self.face_normals) if self.face_normals else np.zeros((0, 3))
        }

    def _compute_face_properties(self):
        self.face_areas = []
        self.face_normals = []

        if len(self.vertices) == 0:
            return

        for face in self.faces:
            n_vert = len(face)
            if n_vert < 3:
                self.face_areas.append(0.0)
                self.face_normals.append([0.0, 0.0, 1.0])
                continue


            verts = []
            for idx in face:
                if 0 <= idx < len(self.vertices):
                    verts.append(self.vertices[idx])
            if len(verts) < 3:
                self.face_areas.append(0.0)
                self.face_normals.append([0.0, 0.0, 1.0])
                continue


            normal = self._newell_normal(verts)
            area = self._polygon_area(verts)

            self.face_normals.append(normal)
            self.face_areas.append(area)

    @staticmethod
    def _newell_normal(vertices: List[List[float]]) -> List[float]:
        n = [0.0, 0.0, 0.0]
        m = len(vertices)
        for i in range(m):
            j = (i + 1) % m
            n[0] += (vertices[i][1] - vertices[j][1]) * (vertices[i][2] + vertices[j][2])
            n[1] += (vertices[i][2] - vertices[j][2]) * (vertices[i][0] + vertices[j][0])
            n[2] += (vertices[i][0] - vertices[j][0]) * (vertices[i][1] + vertices[j][1])

        norm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        if norm > 1e-12:
            n = [n[0]/norm, n[1]/norm, n[2]/norm]
        else:
            n = [0.0, 0.0, 1.0]
        return n

    @staticmethod
    def _polygon_area(vertices: List[List[float]]) -> float:
        if len(vertices) < 3:
            return 0.0

        total_area = 0.0
        v0 = np.array(vertices[0])
        for i in range(1, len(vertices) - 1):
            v1 = np.array(vertices[i])
            v2 = np.array(vertices[i + 1])

            cross = np.cross(v1 - v0, v2 - v0)
            total_area += 0.5 * np.linalg.norm(cross)
        return total_area

    def total_surface_area(self) -> float:
        return float(np.sum(self.face_areas))

    def mean_aperture_estimate(self, volume: float) -> float:
        area = self.total_surface_area()
        if area < 1e-12:
            return 0.0
        return volume / area

    def orientation_distribution(self, n_bins: int = 18) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.face_normals) == 0:
            return np.zeros(n_bins), np.zeros(n_bins)

        normals = np.array(self.face_normals)
        areas = np.array(self.face_areas)



        dips = np.arccos(np.clip(np.abs(normals[:, 2]), -1.0, 1.0))
        dips_deg = np.degrees(dips)

        bins = np.linspace(0, 90, n_bins + 1)
        hist, _ = np.histogram(dips_deg, bins=bins, weights=areas)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        return bin_centers, hist

    def roughness_coefficient(self) -> float:
        if len(self.vertices) == 0:
            return 0.0
        z_coords = self.vertices[:, 2]
        return float(np.std(z_coords))

    def generate_sample_fracture_obj(self, size: float = 1.0, amplitude: float = 0.01,
                                     n_segments: int = 20) -> str:
        lines = ["# Sample fractured surface OBJ"]
        dx = size / n_segments
        dy = size / n_segments


        for i in range(n_segments + 1):
            for j in range(n_segments + 1):
                x = j * dx
                y = i * dy
                z = amplitude * np.sin(2 * np.pi * x / size) * np.cos(2 * np.pi * y / size)
                lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")


        for i in range(n_segments):
            for j in range(n_segments):
                v0 = i * (n_segments + 1) + j + 1
                v1 = v0 + 1
                v2 = (i + 1) * (n_segments + 1) + j + 1
                v3 = v2 + 1
                lines.append(f"f {v0} {v1} {v3}")
                lines.append(f"f {v0} {v3} {v2}")

        return '\n'.join(lines)
