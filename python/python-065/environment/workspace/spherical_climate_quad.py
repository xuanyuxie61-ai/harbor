
import numpy as np


def icosahedron_shape():
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return verts, faces


def sphere01_distance_xyz(xyz1, xyz2):

    diff = xyz1 - xyz2
    chord = np.sqrt(np.sum(diff ** 2))
    return 2.0 * np.arcsin(min(chord / 2.0, 1.0))


def sphere01_triangle_vertices_to_area(a_xyz, b_xyz, c_xyz):
    a = sphere01_distance_xyz(b_xyz, c_xyz)
    b = sphere01_distance_xyz(a_xyz, c_xyz)
    c = sphere01_distance_xyz(a_xyz, b_xyz)

    s = 0.5 * (a + b + c)

    if s <= 0.0 or s >= np.pi:
        return 0.0

    try:
        tan_e4 = np.sqrt(
            max(0.0,
                np.tan(s / 2.0)
                * np.tan((s - a) / 2.0)
                * np.tan((s - b) / 2.0)
                * np.tan((s - c) / 2.0))
        )
    except ValueError:
        return 0.0
    e = 4.0 * np.arctan(tan_e4)
    return max(e, 0.0)


def sphere01_triangle_project(a_xyz, b_xyz, c_xyz, f1, f2, f3):
    v = f1 * np.array(a_xyz) + f2 * np.array(b_xyz) + f3 * np.array(c_xyz)
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.array(a_xyz)
    return v / norm


def sphere01_quad_icos1c(factor, fun):
    if factor < 1:
        factor = 1
    verts, faces = icosahedron_shape()
    result = 0.0
    area_total = 0.0
    node_num = 0

    for face in faces:
        a = verts[face[0]]
        b = verts[face[1]]
        c = verts[face[2]]


        for f3 in range(1, 3 * factor - 1, 3):
            for f2 in range(1, 3 * factor - f3 - 1, 3):
                f1 = 3 * factor - f3 - f2
                node_xyz = sphere01_triangle_project(a, b, c, f1, f2, f3)
                a2 = sphere01_triangle_project(a, b, c, f1 + 2, f2 - 1, f3 - 1)
                b2 = sphere01_triangle_project(a, b, c, f1 - 1, f2 + 2, f3 - 1)
                c2 = sphere01_triangle_project(a, b, c, f1 - 1, f2 - 1, f3 + 2)
                area = sphere01_triangle_vertices_to_area(a2, b2, c2)
                v = fun(node_xyz)
                node_num += 1
                result += area * v
                area_total += area


        for f3 in range(2, 3 * factor - 3, 3):
            for f2 in range(2, 3 * factor - f3 - 2, 3):
                f1 = 3 * factor - f3 - f2
                node_xyz = sphere01_triangle_project(a, b, c, f1, f2, f3)
                a2 = sphere01_triangle_project(a, b, c, f1 - 2, f2 + 1, f3 + 1)
                b2 = sphere01_triangle_project(a, b, c, f1 + 1, f2 - 2, f3 + 1)
                c2 = sphere01_triangle_project(a, b, c, f1 + 1, f2 + 1, f3 - 2)
                area = sphere01_triangle_vertices_to_area(a2, b2, c2)
                v = fun(node_xyz)
                node_num += 1
                result += area * v
                area_total += area

    return result, node_num


def sphere01_sample_3d(n, seed=None):
    rng = np.random.default_rng(seed)
    phi = 2.0 * np.pi * rng.random(n)
    z = 2.0 * rng.random(n) - 1.0
    r = np.sqrt(1.0 - z ** 2)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.vstack([x, y, z]).T


def sphere01_quad_mc(fun, n, seed=None):
    x = sphere01_sample_3d(n, seed)
    v = np.array([fun(x[k]) for k in range(n)])
    result = 4.0 * np.pi * np.mean(v)
    return result


def global_radiation_forcing_integral(forcing_field_latlon, factor=2):
    def fun(xyz):

        lat = np.arcsin(np.clip(xyz[2], -1.0, 1.0))
        lon = np.arctan2(xyz[1], xyz[0])
        return forcing_field_latlon(lat, lon)

    total, node_num = sphere01_quad_icos1c(factor, fun)

    mean_forcing = total / (4.0 * np.pi)
    return mean_forcing, node_num


def test_spherical():

    def fun1(xyz):
        return 1.0
    val, nn = sphere01_quad_icos1c(2, fun1)
    assert abs(val - 4.0 * np.pi) < 0.1
    print("spherical_climate_quad 自测试通过")


if __name__ == "__main__":
    test_spherical()
