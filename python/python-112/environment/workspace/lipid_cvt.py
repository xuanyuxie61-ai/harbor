
import numpy as np
from typing import Tuple, Callable, Optional





def cvt_triangle_uniform(
    triangle: np.ndarray,
    n: int,
    sample_num: int,
    it_num: int,
    density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if triangle.shape != (3, 2):
        raise ValueError("cvt_triangle_uniform: triangle must have shape (3, 2).")
    if n < 3:
        raise ValueError("cvt_triangle_uniform: n must be >= 3.")
    if sample_num < n:
        raise ValueError("cvt_triangle_uniform: sample_num must be >= n.")
    if it_num < 1:
        raise ValueError("cvt_triangle_uniform: it_num must be >= 1.")


    p = _sample_triangle_uniform(triangle, n)

    for it in range(it_num):

        s = _sample_triangle_uniform(triangle, sample_num)



        dists = np.linalg.norm(s[:, None, :] - p[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)


        if density is not None:
            rho_s = density(s)
            rho_s = np.clip(rho_s, 1.0e-12, None)
        else:
            rho_s = np.ones(sample_num, dtype=float)

        p_new = np.zeros_like(p)
        mass = np.zeros(n, dtype=float)
        for i in range(n):
            mask = nearest == i
            count = np.count_nonzero(mask)
            if count > 0:
                mass[i] = np.sum(rho_s[mask])
                p_new[i] = np.sum(s[mask] * rho_s[mask][:, None], axis=0) / mass[i]
            else:

                p_new[i] = _sample_triangle_uniform(triangle, 1)[0]

        p = p_new


    from scipy.spatial import Delaunay
    tri = Delaunay(p)

    return p, tri.simplices


def _sample_triangle_uniform(triangle: np.ndarray, n: int) -> np.ndarray:
    alpha = np.sqrt(np.random.rand(n))
    beta = np.random.rand(n)

    p12 = (1.0 - alpha)[:, None] * triangle[0] + alpha[:, None] * triangle[1]
    p13 = (1.0 - alpha)[:, None] * triangle[0] + alpha[:, None] * triangle[2]

    p = (1.0 - beta)[:, None] * p12 + beta[:, None] * p13
    return p





def cvt_3d_lumping(
    n: int,
    it_num: int,
    s_num: int,
    mu_fun: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    box: Tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 4:
        raise ValueError("cvt_3d_lumping: n must be >= 4.")
    if it_num < 1:
        raise ValueError("cvt_3d_lumping: it_num must be >= 1.")
    if s_num < 2:
        raise ValueError("cvt_3d_lumping: s_num must be >= 2.")

    xmin, xmax, ymin, ymax, zmin, zmax = box


    g = np.zeros((n, 3), dtype=float)
    g[:, 0] = np.random.uniform(xmin, xmax, n)
    g[:, 1] = np.random.uniform(ymin, ymax, n)
    g[:, 2] = np.random.uniform(zmin, zmax, n)


    eps = 1.0e-12
    s_1d_x = np.linspace(xmin + eps, xmax - eps, s_num)
    s_1d_y = np.linspace(ymin + eps, ymax - eps, s_num)
    s_1d_z = np.linspace(zmin + eps, zmax - eps, s_num)

    sx, sy, sz = np.meshgrid(s_1d_x, s_1d_y, s_1d_z, indexing='ij')
    sx_vec = sx.ravel()
    sy_vec = sy.ravel()
    sz_vec = sz.ravel()
    s = np.column_stack((sx_vec, sy_vec, sz_vec))


    mu_mat = mu_fun(sx, sy, sz)
    mu_mat = np.clip(mu_mat, 1.0e-12, 10.0)
    r_vec = mu_mat.ravel() ** 5

    energy = np.full(it_num, np.nan, dtype=float)
    motion = np.full(it_num, np.nan, dtype=float)

    g_new = np.zeros_like(g)

    for it in range(it_num):

        from scipy.spatial import cKDTree
        tree = cKDTree(g)
        k = tree.query(s, k=1)[1]


        m = np.zeros(n, dtype=float)
        g_new[:, 0] = 0.0
        g_new[:, 1] = 0.0
        g_new[:, 2] = 0.0

        for idx in range(n):
            mask = k == idx
            m[idx] = np.sum(r_vec[mask])
            if m[idx] > 0:
                g_new[idx, 0] = np.sum(r_vec[mask] * s[mask, 0]) / m[idx]
                g_new[idx, 1] = np.sum(r_vec[mask] * s[mask, 1]) / m[idx]
                g_new[idx, 2] = np.sum(r_vec[mask] * s[mask, 2]) / m[idx]
            else:

                g_new[idx] = [
                    np.random.uniform(xmin, xmax),
                    np.random.uniform(ymin, ymax),
                    np.random.uniform(zmin, zmax),
                ]


        diff = s - g[k]
        energy[it] = np.sum(r_vec * np.sum(diff ** 2, axis=1)) / s_num


        motion[it] = np.mean(np.sum((g_new - g) ** 2, axis=1))

        g = g_new.copy()

    return g, energy, motion





def place_lipid_bilayer(
    n_lipids_per_leaflet: int = 50,
    protein_radius: float = 15.0,
    box_xy: float = 60.0,
    exclusion_radius: float = 18.0,
    it_num: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_lipids_per_leaflet < 3:
        raise ValueError("place_lipid_bilayer: n_lipids_per_leaflet must be >= 3.")
    if protein_radius < 0 or exclusion_radius < protein_radius:
        raise ValueError("place_lipid_bilayer: exclusion_radius must be > protein_radius >= 0.")

    half = box_xy / 2.0





    def lipid_density(pts: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(pts, axis=1)
        rho = np.ones(pts.shape[0], dtype=float)
        rho[d < exclusion_radius] = 0.0
        return rho


    tri1 = np.array([[0.0, 0.0], [half, 0.0], [half, half]], dtype=float)
    tri2 = np.array([[0.0, 0.0], [half, half], [0.0, half]], dtype=float)
    tri3 = np.array([[0.0, 0.0], [0.0, -half], [half, -half]], dtype=float)
    tri4 = np.array([[0.0, 0.0], [half, -half], [half, 0.0]], dtype=float)

    def run_cvt_for_triangle(tri, n_sub):
        p, _ = cvt_triangle_uniform(
            tri, n_sub, sample_num=2000 * n_sub, it_num=it_num, density=lipid_density
        )
        return p

    n_sub = max(3, n_lipids_per_leaflet // 4)
    p1 = run_cvt_for_triangle(tri1, n_sub)
    p2 = run_cvt_for_triangle(tri2, n_sub)
    p3 = run_cvt_for_triangle(tri3, n_sub)
    p4 = run_cvt_for_triangle(tri4, n_sub)


    upper_all = np.vstack([p1, p2, p3, p4])

    mask = np.linalg.norm(upper_all, axis=1) >= exclusion_radius
    upper_all = upper_all[mask]


    if upper_all.shape[0] > n_lipids_per_leaflet:
        idx = np.random.choice(upper_all.shape[0], n_lipids_per_leaflet, replace=False)
        upper_leaflet = upper_all[idx]
    else:
        upper_leaflet = upper_all


    lower_leaflet = upper_leaflet.copy()

    upper_z = 15.0
    lower_z = -15.0

    return upper_leaflet, lower_leaflet, upper_z, lower_z
