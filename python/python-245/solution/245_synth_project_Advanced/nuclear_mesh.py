"""
nuclear_mesh.py
===============
Mesh generation and management for the 3D representation of a fissioning nucleus.

Maps from: 382_fem_to_xml (mesh format conversion),
           872_ply_display (PLY 3D mesh with face triangulation)

Physical context
----------------
The fissioning nucleus is represented by a shape parametrisation in cylindrical
coordinates (r, z).  For the Langevin dynamics approach, we track the nuclear
surface on a 3D mesh.  The mesh evolves from a single sphere (compound nucleus)
through scission into two separate fragment shapes.

We implement:
1. Triangular mesh generation on a parameterised nuclear surface
2. Face triangulation for arbitrary polygon faces (from PLY concept)
3. Mesh serialisation to an XML-like structure (from fem_to_xml concept)
4. Volume and surface area computation via divergence theorem
"""

import math
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# 3D vertex type
# ---------------------------------------------------------------------------
Vertex3D = Tuple[float, float, float]
Triangle = Tuple[int, int, int]


class NuclearMesh:
    """
    Triangular surface mesh representing a fissioning nuclear shape.

    The shape is parameterised by (c, h, alpha) deformation parameters
    in the Brack-Damgaard framework:
      - c : elongation parameter (c=1 is sphere)
      - h : neck parameter
      - alpha : mass asymmetry

    The nuclear surface in cylindrical coords (rho, z) is:
      rho_s(z)^2 = R_0^2 * [c^2 * u^2 * (1 - h + alpha*u + (1-h)*u^2)]
                   if rho_s^2 > 0, else rho_s = 0 (two fragments)

    where u = z / (c * R_0) and R_0 = r0 * A^{1/3}.
    """

    def __init__(self, mass_number: int = 236, n_z_sections: int = 20,
                 n_phi_sections: int = 16):
        self.A = mass_number
        self.nz = max(4, n_z_sections)
        self.nphi = max(4, n_phi_sections)
        self.vertices: List[Vertex3D] = []
        self.faces: List[Tuple[int, ...]] = []  # may start as polygons
        self.triangles: List[Triangle] = []      # after triangulation
        self.r0 = 1.25  # fm, nuclear radius parameter
        self._compute_r0()

    def _compute_r0(self) -> None:
        """R_0 = r0 * A^{1/3} in fm (volume-equivalent sphere radius)."""
        self.r0 = 1.25 * (self.A ** (1.0 / 3.0))

    def generate_shape(self, c: float, h: float, alpha: float) -> None:
        """
        Generate mesh for shape parameters (c, h, alpha).

        The nuclear surface is:
          rho_s(z)^2 = R0^2 * c^2 * u^2 * sigma(u)
        where
          sigma(u) = 1 - h + alpha*u + (1-h)*u^2
          u = z / (c*R0),  z in [-c*R0, c*R0]

        Volume conservation requires c^3 * (1 + 2/5*(1-h)) = 1 approximately,
        but for this small-scale experiment we enforce volume conservation
        by adjusting c if needed.
        """
        # Volume-conservation check
        # V ~ (4/3)*pi*R0^3 * c^3 * (1 + (2/5)*(1-h))
        # For V = (4/3)*pi*R0^3: c^3*(1+(2/5)*(1-h)) = 1
        # We solve for c if h is given
        vol_factor = 1.0 + 0.4 * (1.0 - h)
        if vol_factor > 0:
            c_vc = c * (vol_factor ** (-1.0 / 3.0))
        else:
            c_vc = c

        r0 = self.r0
        half_z = c_vc * r0
        dz = 2.0 * half_z / self.nz

        self.vertices = []
        polygonal_faces = []

        for iz in range(self.nz + 1):
            z_coord = -half_z + iz * dz
            u = z_coord / (c_vc * r0) if c_vc * r0 > 1e-10 else 0.0

            # Shape function sigma(u)
            sigma = 1.0 - h + alpha * u + (1.0 - h) * u * u
            # Clamp to prevent negative rho^2
            sigma = max(sigma, 0.0)
            rho_s = r0 * c_vc * abs(u) * math.sqrt(sigma) if sigma > 0 else 0.0

            # Create ring of vertices at this z
            ring_start = len(self.vertices)
            if rho_s < 1e-12:
                # Single point on axis
                self.vertices.append((0.0, 0.0, z_coord))
                ring_size = 1
            else:
                ring_size = self.nphi
                dphi = 2.0 * math.pi / self.nphi
                for iphi in range(self.nphi):
                    phi = iphi * dphi
                    x = rho_s * math.cos(phi)
                    y = rho_s * math.sin(phi)
                    self.vertices.append((x, y, z_coord))

            polygonal_faces.append((ring_start, ring_size))

        # Build faces connecting adjacent rings
        self.faces = []
        for iz in range(len(polygonal_faces) - 1):
            rs1, sz1 = polygonal_faces[iz]
            rs2, sz2 = polygonal_faces[iz + 1]

            if sz1 == 1 and sz2 == 1:
                continue  # skip degenerate
            elif sz1 == 1:
                # Fan from single vertex to ring 2
                for j in range(sz2):
                    j_next = (j + 1) % sz2
                    self.faces.append((rs1, rs2 + j, rs2 + j_next))
            elif sz2 == 1:
                # Fan from ring 1 to single vertex
                for j in range(sz1):
                    j_next = (j + 1) % sz1
                    self.faces.append((rs1 + j, rs1 + j_next, rs2))
            else:
                # Quad strips -> pairs of triangles
                for j in range(max(sz1, sz2)):
                    j1 = j % sz1
                    j1n = (j + 1) % sz1
                    j2 = j % sz2
                    j2n = (j + 1) % sz2
                    # Quad: (rs1+j1, rs1+j1n, rs2+j2n, rs2+j2)
                    # Split into two triangles
                    self.faces.append((rs1 + j1, rs1 + j1n, rs2 + j2))
                    self.faces.append((rs1 + j1n, rs2 + j2n, rs2 + j2))

        self._triangulate_faces()

    def _triangulate_faces(self) -> None:
        """
        Triangulate polygon faces using fan triangulation (from PLY concept).
        For n-gon face (v1, v2, ..., vn), create triangles (v1, v_{j-1}, v_j)
        for j = 3..n.
        """
        self.triangles = []
        for face in self.faces:
            n = len(face)
            if n == 3:
                self.triangles.append(face)
            elif n > 3:
                for j in range(2, n):
                    self.triangles.append((face[0], face[j - 1], face[j]))

    def compute_volume(self) -> float:
        """
        Compute enclosed volume using divergence theorem:
          V = (1/3) * sum_triangles (A_face . n_hat * r_centroid)

        For triangular mesh: V = (1/6) * |sum_tri v0 . (v1 x v2)|
        (signed volume of tetrahedra from origin)
        """
        vol = 0.0
        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            # Scalar triple product v0 . (v1 x v2)
            cross_x = v1[1] * v2[2] - v1[2] * v2[1]
            cross_y = v1[2] * v2[0] - v1[0] * v2[2]
            cross_z = v1[0] * v2[1] - v1[1] * v2[0]
            vol += v0[0] * cross_x + v0[1] * cross_y + v0[2] * cross_z
        return abs(vol) / 6.0

    def compute_surface_area(self) -> float:
        """
        Surface area = sum of triangle areas.
        Area_i = (1/2) * |e1 x e2| where e1 = v1-v0, e2 = v2-v0.
        """
        area = 0.0
        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            # Cross product
            cx = e1[1] * e2[2] - e1[2] * e2[1]
            cy = e1[2] * e2[0] - e1[0] * e2[2]
            cz = e1[0] * e2[1] - e1[1] * e2[0]
            area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        return area

    def export_xml(self) -> str:
        """
        Export mesh to DOLFIN-compatible XML format (from fem_to_xml concept).
        Generates mesh with triangle cells in 3D.
        """
        lines = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(f'<dolfin xmlns:dolfin="https://fenicsproject.org/">')
        lines.append(f'  <mesh celltype="triangle" dim="3">')
        # Vertices
        lines.append(f'    <vertices size="{len(self.vertices)}">')
        for i, v in enumerate(self.vertices):
            lines.append(
                f'      <vertex index="{i}" x="{v[0]:.8f}" '
                f'y="{v[1]:.8f}" z="{v[2]:.8f}"/>'
            )
        lines.append('    </vertices>')
        # Cells (triangles)
        lines.append(f'    <cells size="{len(self.triangles)}">')
        for i, tri in enumerate(self.triangles):
            lines.append(
                f'      <triangle index="{i}" '
                f'v0="{tri[0]}" v1="{tri[1]}" v2="{tri[2]}"/>'
            )
        lines.append('    </cells>')
        lines.append('  </mesh>')
        lines.append('</dolfin>')
        return '\n'.join(lines)

    def summary(self) -> str:
        """Return a text summary of the mesh."""
        vol = self.compute_volume()
        area = self.compute_surface_area()
        # Volume of equivalent sphere: V = (4/3)*pi*R^3
        r_equiv = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0) if vol > 0 else 0.0
        lines = [
            f"NuclearMesh: A={self.A}, nz={self.nz}, nphi={self.nphi}",
            f"  Vertices: {len(self.vertices)}",
            f"  Triangles: {len(self.triangles)}",
            f"  Volume: {vol:.4f} fm^3",
            f"  Surface area: {area:.4f} fm^2",
            f"  Equivalent sphere radius: {r_equiv:.4f} fm",
        ]
        return '\n'.join(lines)


def barycentric_coordinates(p: Vertex3D, v0: Vertex3D,
                            v1: Vertex3D, v2: Vertex3D) -> Tuple[float, float, float]:
    """
    Compute barycentric coordinates of point p with respect to triangle (v0, v1, v2).
    Uses the area-ratio method:
      lambda_i = Area(p, v_j, v_k) / Area(v0, v1, v2)
    (from 417_fem3d_pack tetrahedron_barycentric concept adapted to 2D surface)
    """
    # Edge vectors
    e0 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e1 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    e2 = (p[0] - v0[0], p[1] - v0[1], p[2] - v0[2])

    d00 = e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]
    d01 = e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2]
    d11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]
    d20 = e2[0] * e0[0] + e2[1] * e0[1] + e2[2] * e0[2]
    d21 = e2[0] * e1[0] + e2[1] * e1[1] + e2[2] * e1[2]

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    lam1 = (d11 * d20 - d01 * d21) / denom
    lam2 = (d00 * d21 - d01 * d20) / denom
    lam0 = 1.0 - lam1 - lam2
    return (lam0, lam1, lam2)
