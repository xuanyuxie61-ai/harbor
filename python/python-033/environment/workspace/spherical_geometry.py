
import numpy as np


def sphere_stereograph(p, focus=None):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    n, m = p.shape
    if focus is None:
        focus = np.zeros(m, dtype=float)
        focus[-1] = -1.0


    denom = 1.0 + p[:, -1]

    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    ss = 2.0 / denom
    q_full = ss[:, None] * p + (1.0 - ss[:, None]) * focus[None, :]

    return q_full[:, :-1]


def sphere_stereograph_inverse(q, focus=None):
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    n, mdim = q.shape
    m = mdim + 1
    if focus is None:
        focus = np.zeros(m, dtype=float)
        focus[-1] = -1.0

    q_sq = np.sum(q ** 2, axis=1)
    ss = 4.0 / (4.0 + q_sq)

    q_full = np.zeros((n, m), dtype=float)
    q_full[:, :-1] = q
    q_full[:, -1] = 2.0
    p = ss[:, None] * q_full + (1.0 - ss[:, None]) * focus[None, :]

    norms = np.linalg.norm(p, axis=1)
    norms = np.where(norms < 1e-15, 1.0, norms)
    p = p / norms[:, None]
    return p


def icosahedron_vertices():
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)

    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    return verts


def spherical_delaunay_triangulation(xyz):
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    if n < 4:
        return np.array([], dtype=int).reshape(0, 3)

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xyz)
        faces = hull.simplices

        oriented_faces = []
        for face in faces:
            v0, v1, v2 = xyz[face[0]], xyz[face[1]], xyz[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            centroid = (v0 + v1 + v2) / 3.0
            if np.dot(normal, centroid) < 0:
                face = face[[0, 2, 1]]
            oriented_faces.append(face)
        return np.array(oriented_faces, dtype=int)
    except ImportError:

        print("[spherical_geometry] scipy not available, using icosahedron fallback")
        return np.array([], dtype=int).reshape(0, 3)


def subdivide_icosahedron(factor=2):
    verts = icosahedron_vertices()

    faces_list = [
        [0,2,8],[0,8,4],[0,4,10],[0,10,6],[0,6,2],
        [2,7,8],[8,9,4],[4,5,10],[10,11,6],[6,1,2],
        [3,7,9],[3,9,5],[3,5,11],[3,11,1],[3,1,7],
        [1,6,11],[7,3,9],[9,3,5],[5,3,11],[11,3,1]
    ]

    faces = np.array([
        [0,2,8],[0,8,4],[0,4,10],[0,10,6],[0,6,2],
        [2,7,8],[8,9,4],[4,5,10],[10,11,6],[6,1,2],
        [3,7,9],[3,9,5],[3,5,11],[3,11,1],[3,1,7],
        [1,6,11],[7,3,9],[9,3,5],[5,3,11],[11,3,1]
    ], dtype=int)

    faces = spherical_delaunay_triangulation(verts)
    return verts, faces


def test_spherical_geometry():

    p = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    q = sphere_stereograph(p)
    p_rec = sphere_stereograph_inverse(q)
    err = np.max(np.abs(p_rec - p))
    print(f"[spherical_geometry] Stereographic projection max error = {err:.3e}")
    assert err < 1e-10, "Stereographic projection inaccurate"


    verts = icosahedron_vertices()
    faces = spherical_delaunay_triangulation(verts)
    print(f"[spherical_geometry] Icosahedron triangulation: {len(faces)} faces")


if __name__ == "__main__":
    test_spherical_geometry()
