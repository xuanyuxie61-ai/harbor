# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional


class CompositePly:

    def __init__(self, thickness: float, fiber_angle: float,
                 E1: float, E2: float, G12: float, nu12: float,
                 Vf: float, density: float):
        if thickness <= 0:
            raise ValueError("Ply thickness must be positive.")
        if not (0.0 <= Vf <= 1.0):
            raise ValueError("Fiber volume fraction Vf must be in [0,1].")
        if E1 <= 0 or E2 <= 0 or G12 <= 0:
            raise ValueError("Elastic moduli must be positive.")

        self.thickness = thickness
        self.fiber_angle = np.radians(fiber_angle)
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.Vf = Vf
        self.density = density



        nu21 = nu12 * E2 / E1
        denom = 1.0 - nu12 * nu21
        if abs(denom) < 1e-15:
            raise ValueError("Invalid Poisson ratio combination leads to singularity.")

        Q11 = E1 / denom
        Q12 = nu12 * E2 / denom
        Q22 = E2 / denom
        Q66 = G12

        c = np.cos(self.fiber_angle)
        s = np.sin(self.fiber_angle)
        c2 = c * c
        s2 = s * s
        c4 = c2 * c2
        s4 = s2 * s2


        self.Qbar = np.zeros((3, 3))
        self.Qbar[0, 0] = Q11 * c4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * s4
        self.Qbar[0, 1] = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (s4 + c4)
        self.Qbar[1, 0] = self.Qbar[0, 1]
        self.Qbar[1, 1] = Q11 * s4 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * c4
        self.Qbar[0, 2] = (Q11 - Q12 - 2 * Q66) * c * s * c2 + (Q12 - Q22 + 2 * Q66) * c * s * s2
        self.Qbar[2, 0] = self.Qbar[0, 2]
        self.Qbar[1, 2] = (Q11 - Q12 - 2 * Q66) * c * s * s2 + (Q12 - Q22 + 2 * Q66) * c * s * c2
        self.Qbar[2, 1] = self.Qbar[1, 2]
        self.Qbar[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (s4 + c4)


        self.E_eff = self.Qbar[0, 0]


class CompositeLaminate:

    def __init__(self, plies: List[CompositePly]):
        if not plies:
            raise ValueError("At least one ply is required.")
        self.plies = plies
        self.num_plies = len(plies)
        self.total_thickness = sum(p.thickness for p in plies)


        self.z_mids = np.zeros(self.num_plies)
        z = -self.total_thickness / 2.0
        for i, p in enumerate(plies):
            z += p.thickness / 2.0
            self.z_mids[i] = z
            z += p.thickness / 2.0


        self.rho_eff = sum(p.density * p.thickness for p in plies) / self.total_thickness

        self.E_eff = sum(p.E_eff * p.thickness for p in plies) / self.total_thickness

    def get_homogenized_properties(self) -> dict:
        return {
            "rho_eff": self.rho_eff,
            "E_eff": self.E_eff,
            "h_total": self.total_thickness,
            "num_plies": self.num_plies,
        }


class Mesh1D:

    def __init__(self, x_min: float, x_max: float, num_elements: int,
                 refine_strength: float = 0.0, refine_center: Optional[float] = None):
        if num_elements < 2:
            raise ValueError("num_elements must be >= 2.")
        if x_max <= x_min:
            raise ValueError("x_max must be > x_min.")
        if refine_strength < 0:
            raise ValueError("refine_strength must be non-negative.")

        self.x_min = x_min
        self.x_max = x_max
        self.L = x_max - x_min
        self.num_elements = num_elements
        self.refine_strength = refine_strength
        self.refine_center = refine_center if refine_center is not None else (x_min + x_max) / 2.0


        self.nodes = self._generate_nodes()
        self.elements = self._build_elements()
        self.element_sizes = self.nodes[1:] - self.nodes[:-1]


        self.interface_flags = np.zeros(num_elements, dtype=bool)

    def _generate_nodes(self) -> np.ndarray:
        xi = np.linspace(0.0, 1.0, self.num_elements + 1)
        a = self.refine_strength

        f_xi = xi + a * 0.5 * (1.0 - np.cos(np.pi * xi))
        f_1 = 1.0 + a * 0.5 * (1.0 - np.cos(np.pi))
        x = self.x_min + self.L * f_xi / f_1

        x[0] = self.x_min
        x[-1] = self.x_max

        if np.any(np.diff(x) <= 0):
            raise RuntimeError("Mesh generation failed: non-monotonic nodes detected.")
        return x

    def _build_elements(self) -> List[Tuple[int, int]]:
        elements = []
        for i in range(self.num_elements):
            elements.append((i, i + 1))
        return elements

    def get_element_centers(self) -> np.ndarray:
        return 0.5 * (self.nodes[:-1] + self.nodes[1:])

    def get_element_jacobians(self) -> np.ndarray:
        return self.element_sizes / 2.0

    def locate_point(self, x: float) -> int:
        if x < self.x_min or x > self.x_max:
            raise ValueError(f"Point x={x} out of domain [{self.x_min}, {self.x_max}].")
        if x >= self.x_max:
            return self.num_elements - 1

        left, right = 0, self.num_elements
        while left < right:
            mid = (left + right) // 2
            if self.nodes[mid] <= x < self.nodes[mid + 1]:
                return mid
            elif x < self.nodes[mid]:
                right = mid
            else:
                left = mid + 1
        return min(left, self.num_elements - 1)


def build_default_laminate() -> CompositeLaminate:

    E1 = 181.0e9
    E2 = 10.3e9
    G12 = 7.17e9
    nu12 = 0.28
    rho = 1600.0
    Vf = 0.62
    h_ply = 0.125e-3

    angles = [0, 45, -45, 90, 90, -45, 45, 0]
    plies = []
    for theta in angles:
        plies.append(CompositePly(
            thickness=h_ply,
            fiber_angle=theta,
            E1=E1, E2=E2, G12=G12, nu12=nu12,
            Vf=Vf, density=rho
        ))
    return CompositeLaminate(plies)


def build_default_mesh(L: float = 1.0, num_elements: int = 40,
                       refine_strength: float = 1.5) -> Mesh1D:
    return Mesh1D(x_min=0.0, x_max=L, num_elements=num_elements,
                  refine_strength=refine_strength, refine_center=L / 2.0)


if __name__ == "__main__":

    laminate = build_default_laminate()
    props = laminate.get_homogenized_properties()
    print("Laminate properties:", props)

    mesh = build_default_mesh()
    print("Mesh nodes (first 5):", mesh.nodes[:5])
    print("Element sizes (first 5):", mesh.element_sizes[:5])
    print("Element centers (first 5):", mesh.get_element_centers()[:5])
    print("Locate 0.5:", mesh.locate_point(0.5))
