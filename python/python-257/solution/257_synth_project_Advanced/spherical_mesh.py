"""
spherical_mesh.py
=================
Cubed-sphere and icosahedral pixelisation of S^2  for the CMB sky map.

Two pixelisations are supported:
  * CubedSphere   – gnomonic projection of a face-centred cube onto S^2.
  * Icosahedral   – recursive subdivision of an icosahedron.

The key geometric objects needed by the CMB pipeline are:
  - node coordinates  (theta, phi) in radians
  - adjacency lists   (for FD stencils)
  - solid angles      (for quadrature)
  - element / face connectivity  (for FEM assembly)

Reference mesh topology is taken from triangulation_svg and
fem_basis_t4_display (cubic-bubble elements).
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Cubed-sphere pixelisation
# ---------------------------------------------------------------------------
class CubedSphere:
    """
    Nside-parameterised cubed-sphere mesh.
    Nside = number of cells per edge of one cube face.
    Total number of nodes  = 6 * Nside^2  (centre of each pixel)
    Total number of faces  = 6 * Nside^2
    """

    FACE_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]

    def __init__(self, nside: int):
        if nside < 1:
            raise ValueError("nside must be >= 1")
        self.nside = nside
        self.nodes: List[Tuple[float, float]] = []       # (theta, phi)
        self.cartesian: List[Tuple[float, float, float]] = []
        self.faces: List[Tuple[int, int, int, int]] = [] # quad faces
        self.neighbours: Dict[int, List[int]] = {}
        self._build()

    # ----- construction ---------------------------------------------------
    def _build(self) -> None:
        N = self.nside
        # Face centres on the unit cube (±1, ±1, ±1 projected)
        for face in range(6):
            ix = (face >> 1) & 1   # which of {x,y,z}
            iy = (face >> 0) & 1
            for j in range(N):
                for k in range(N):
                    u = (j + 0.5) / N * 2.0 - 1.0
                    v = (k + 0.5) / N * 2.0 - 1.0
                    x, y, z = self._face_to_xyz(face, u, v)
                    self.cartesian.append((x, y, z))
                    self.nodes.append(self._xyz_to_thetaphi(x, y, z))

        # Faces (quads): connect neighbouring pixel centres
        self.faces = self._build_faces()
        self.neighbours = self._build_neighbours()

    def _face_to_xyz(self, face: int, u: float, v: float) -> Tuple[float, float, float]:
        """Gnomonic projection from a face (u,v) in [-1,1]^2 to unit S^2."""
        face_idx = face % 6
        if face_idx == 0:     # +x face
            x, y, z = 1.0, u, v
        elif face_idx == 1:   # -x face
            x, y, z = -1.0, -u, v
        elif face_idx == 2:   # +y face
            x, y, z = -u, 1.0, v
        elif face_idx == 3:   # -y face
            x, y, z = u, -1.0, v
        elif face_idx == 4:   # +z face
            x, y, z = -u, v, 1.0
        else:                 # -z face
            x, y, z = -u, -v, -1.0
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1.0e-30:
            raise ValueError("Degenerate face projection.")
        return x / norm, y / norm, z / norm

    @staticmethod
    def _xyz_to_thetaphi(x: float, y: float, z: float) -> Tuple[float, float]:
        """Cartesian -> (theta in [0,pi], phi in [0,2pi))."""
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1.0e-30:
            return 0.0, 0.0
        theta = math.acos(max(-1.0, min(1.0, z / r)))
        phi = math.atan2(y, x)
        if phi < 0.0:
            phi += 2.0 * math.pi
        return theta, phi

    def _build_faces(self) -> List[Tuple[int, int, int, int]]:
        N = self.nside
        faces: List[Tuple[int, int, int, int]] = []
        for face in range(6):
            for j in range(N):
                for k in range(N):
                    idx = face * N * N + j * N + k
                    # local neighbours within face
                    iR = idx + 1 if (j * N + k + 1) < N * N and (k + 1) < N else -1
                    iT = idx + N if j + 1 < N else -1
                    iRT = idx + N + 1 if iT >= 0 and iR >= 0 else -1
                    if iR >= 0 and iT >= 0 and iRT >= 0:
                        faces.append((idx, iR, iRT, iT))
        return faces

    def _build_neighbours(self) -> Dict[int, List[int]]:
        """Build 4-connected neighbour list via shared faces."""
        nbrs: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes)}
        for f in self.faces:
            for i in range(4):
                a, b = f[i], f[(i + 1) % 4]
                if a not in nbrs[b]:
                    nbrs[b].append(a)
                if b not in nbrs[a]:
                    nbrs[a].append(b)
        return nbrs

    # ----- public accessors -----------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def solid_angles(self) -> List[float]:
        """Approximate equal-area solid angles  ~ 4 pi / n_nodes."""
        omega = 4.0 * math.pi / self.n_nodes
        return [omega] * self.n_nodes

    def angular_separation(self, i: int, j: int) -> float:
        """Great-circle distance between node i and node j (radians)."""
        xi, yi, zi = self.cartesian[i]
        xj, yj, zj = self.cartesian[j]
        dot = max(-1.0, min(1.0, xi * xj + yi * yj + zi * zj))
        return math.acos(dot)

    def neighbour_distances(self, i: int) -> List[float]:
        """Angular separations to every neighbour of node i."""
        return [self.angular_separation(i, j) for j in self.neighbours[i]]

    def mean_pixel_spacing(self) -> float:
        """Mean angular separation between 4-connected neighbours (radians)."""
        tot = 0.0
        cnt = 0
        for i in range(self.n_nodes):
            for j in self.neighbours[i]:
                tot += self.angular_separation(i, j)
                cnt += 1
        return tot / max(1, cnt)


# ---------------------------------------------------------------------------
# Icosahedral mesh (20-face base, recursively subdivided)
# ---------------------------------------------------------------------------
class IcosahedralMesh:
    """
    Recursive icosahedral subdivision.
    level = 0 -> 12 nodes, 20 faces
    level = 1 -> 42 nodes, 80 faces
    level = 2 -> 162 nodes, 320 faces
    """

    PHI = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio

    def __init__(self, level: int):
        if level < 0 or level > 4:
            raise ValueError("level must be in [0, 4].")
        self.level = level
        self.nodes: List[Tuple[float, float, float]] = []
        self.faces: List[Tuple[int, int, int]] = []
        self.neighbours: Dict[int, List[int]] = {}
        self.thetaphi: List[Tuple[float, float]] = []
        self._build()

    def _build(self) -> None:
        p = self.PHI
        # 12 vertices of icosahedron
        raw = [
            (-1,  p,  0), ( 1,  p,  0), (-1, -p,  0), ( 1, -p,  0),
            ( 0, -1,  p), ( 0,  1,  p), ( 0, -1, -p), ( 0,  1, -p),
            ( p,  0, -1), ( p,  0,  1), (-p,  0, -1), (-p,  0,  1),
        ]
        self.nodes = [self._normalise(x, y, z) for (x, y, z) in raw]
        # 20 faces
        self.faces = [
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
        ]
        for _ in range(self.level):
            self._subdivide()

        # precompute spherical coords
        self.thetaphi = [CubedSphere._xyz_to_thetaphi(x, y, z) for (x, y, z) in self.nodes]
        self._build_neighbours()

    def _subdivide(self) -> None:
        edge_midpoints: Dict[Tuple[int, int], int] = {}
        new_faces: List[Tuple[int, int, int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in edge_midpoints:
                return edge_midpoints[key]
            xa, ya, za = self.nodes[a]
            xb, yb, zb = self.nodes[b]
            m = self._normalise((xa + xb) / 2, (ya + yb) / 2, (za + zb) / 2)
            idx = len(self.nodes)
            self.nodes.append(m)
            edge_midpoints[key] = idx
            return idx

        for (a, b, c) in self.faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        self.faces = new_faces

    def _build_neighbours(self) -> None:
        nbrs: Dict[int, List[int]] = {i: [] for i in range(len(self.nodes))}
        for (a, b, c) in self.faces:
            for x, y in [(a, b), (b, c), (c, a)]:
                if y not in nbrs[x]:
                    nbrs[x].append(y)
                if x not in nbrs[y]:
                    nbrs[y].append(x)
        self.neighbours = nbrs

    @staticmethod
    def _normalise(x: float, y: float, z: float) -> Tuple[float, float, float]:
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1.0e-30:
            return (0.0, 0.0, 1.0)
        return (x / r, y / r, z / r)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def solid_angles(self) -> List[float]:
        omega = 4.0 * math.pi / self.n_nodes
        return [omega] * self.n_nodes


# ---------------------------------------------------------------------------
# Utility: great-circle distance
# ---------------------------------------------------------------------------
def great_circle_distance(theta1: float, phi1: float,
                           theta2: float, phi2: float) -> float:
    """Vincenty formula for great-circle distance on the unit sphere."""
    dphi = phi2 - phi1
    num = math.sqrt(
        (math.cos(theta2) * math.sin(dphi)) ** 2 +
        (math.cos(theta1) * math.sin(theta2) - math.sin(theta1) * math.cos(theta2) * math.cos(dphi)) ** 2
    )
    den = (math.sin(theta1) * math.sin(theta2) +
           math.cos(theta1) * math.cos(theta2) * math.cos(dphi))
    return math.atan2(num, den)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cs = CubedSphere(nside=4)
    print(f"CubedSphere(nside=4): {cs.n_nodes} nodes, {cs.n_faces} faces")
    print(f"  mean spacing = {cs.mean_pixel_spacing()*180/math.pi:.2f} deg")
    im = IcosahedralMesh(level=2)
    print(f"IcosahedralMesh(level=2): {im.n_nodes} nodes, {im.n_faces} faces")
    print(f"  node 0 neighbours = {len(im.neighbours[0])}")
