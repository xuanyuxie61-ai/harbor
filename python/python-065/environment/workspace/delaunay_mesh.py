
import numpy as np


def triangle_area_2d(t):
    area = 0.5 * abs(
        t[0, 0] * (t[1, 1] - t[1, 2])
        + t[0, 1] * (t[1, 2] - t[1, 0])
        + t[0, 2] * (t[1, 0] - t[1, 1])
    )
    return area


def points_delaunay_naive_2d(node_xy):
    node_num = node_xy.shape[1]
    if node_num < 3:
        return 0, np.zeros((3, 0), dtype=np.int64)

    z = node_xy[0, :] ** 2 + node_xy[1, :] ** 2
    triangle_num = 0
    triangles = []

    for i in range(node_num - 2):
        for j in range(i + 1, node_num):
            for k in range(i + 1, node_num):
                if j == k:
                    continue
                xn = (node_xy[1, j] - node_xy[1, i]) * (z[k] - z[i]) \
                     - (node_xy[1, k] - node_xy[1, i]) * (z[j] - z[i])
                yn = (node_xy[0, k] - node_xy[0, i]) * (z[j] - z[i]) \
                     - (node_xy[0, j] - node_xy[0, i]) * (z[k] - z[i])
                zn = (node_xy[0, j] - node_xy[0, i]) * (node_xy[1, k] - node_xy[1, i]) \
                     - (node_xy[0, k] - node_xy[0, i]) * (node_xy[1, j] - node_xy[1, i])

                flag = zn < 0.0
                if flag:
                    for m in range(node_num):
                        val = (node_xy[0, m] - node_xy[0, i]) * xn \
                              + (node_xy[1, m] - node_xy[1, i]) * yn \
                              + (z[m] - z[i]) * zn
                        if val > 0.0:
                            flag = False
                            break
                if flag:
                    triangles.append([i, j, k])
                    triangle_num += 1

    if triangle_num == 0:
        return 0, np.zeros((3, 0), dtype=np.int64)
    triangle_node = np.array(triangles, dtype=np.int64).T
    return triangle_num, triangle_node


def circumcenter_2d(a, b, c):
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-14:
        return None
    ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
          + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
          + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
    uy = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
          + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
          + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d
    return np.array([ux, uy])


def build_event_mesh(component_mask, grid_x, grid_y):


    raise NotImplementedError("极端事件网格构建待实现")


def test_delaunay():
    pts = np.array([[0.0, 1.0, 0.5],
                    [0.0, 0.0, 1.0]])
    n, tri = points_delaunay_naive_2d(pts)
    assert n == 1
    print("delaunay_mesh 自测试通过")


if __name__ == "__main__":
    test_delaunay()
